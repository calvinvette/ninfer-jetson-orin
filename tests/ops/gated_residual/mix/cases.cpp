#include "ops/suite.h"
#include "support.h"
#include "core/decode_graph.h"

namespace {
using namespace ninfer::test;
using namespace ninfer;
using namespace ninfer::test::hc::mix;

int run(Weights& weights, std::span<const float> catalog, std::span<const Column> ref, int m,
        CudaExecutionView execution, bool captured = false) {
    Fixture f(weights, m);
    f.upload(catalog, execution.stream());
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (captured) {
        definition.capture(execution.stream(), [&] { f.invoke(execution); });
        graph.instantiate(definition);
        graph.launch(execution.stream());
    } else
        f.invoke(execution);
    ninfer::test::cuda_synchronize(execution.stream());
    const auto label = std::string(name(weights.kind)) + " M=" + std::to_string(m) +
                       " pattern=" + std::to_string(int(weights.pattern)) +
                       " graph=" + std::to_string(captured);
    int failed = verify(f, ref, label) + f.check(label);
    if (captured) {
        const auto next     = input_catalog(weights.pattern, 1);
        const auto next_ref = reference(weights, next);
        f.upload(next, execution.stream());
        graph.launch(execution.stream());
        ninfer::test::cuda_synchronize(execution.stream());
        failed += verify(f, next_ref, label + " changed input") + f.check(label + " changed input");
    }
    return failed;
}

int serial_column_parity() {
    Weights weights(Kind::Block, Pattern::Dense);
    const auto catalog = input_catalog(Pattern::Dense);
    const auto ref     = reference(weights, catalog);
    const auto cuda    = cuda_environment().context.execution_view();
    Fixture batched(weights, 5), serial(weights, 1);
    batched.upload(catalog, cuda.stream());
    batched.invoke(cuda);
    cuda_synchronize(cuda.stream());
    int failed = verify(batched, ref, "block-mix serial-column parity M=5");
    const auto batch_output = from_device<std::uint16_t>(batched.x.data, std::size_t(kWidth) * 5);
    const auto batch_injection = from_device<float>(batched.j.data, std::size_t(kStreams) * 5);
    for (int column = 0; column < 5; ++column) {
        const auto input = std::span{catalog}.subspan(std::size_t(column) * kTotal, kTotal);
        serial.upload(input, cuda.stream());
        serial.invoke(cuda);
        cuda_synchronize(cuda.stream());
        failed += verify(serial, std::span{&ref[column], 1},
                         "block-mix serial-column oracle column=" + std::to_string(column));
        const auto serial_output = from_device<std::uint16_t>(serial.x.data, kWidth);
        const auto serial_injection = from_device<float>(serial.j.data, kStreams);
        const std::vector<std::uint16_t> expected_output(
            batch_output.begin() + std::size_t(column) * kWidth,
            batch_output.begin() + std::size_t(column + 1) * kWidth);
        const std::vector<float> expected_injection(
            batch_injection.begin() + std::size_t(column) * kStreams,
            batch_injection.begin() + std::size_t(column + 1) * kStreams);
        failed += verify_exact(("block-mix serial output column=" + std::to_string(column)).c_str(),
                               serial_output, expected_output);
        failed += verify_exact(
            ("block-mix serial injection column=" + std::to_string(column)).c_str(),
            serial_injection, expected_injection);
        failed += serial.check("block-mix serial-column parity");
    }
    return failed + batched.check("block-mix batched serial-column parity");
}

} // namespace

namespace ninfer::test::hc {
void add_mix_cases(OpSuite& suite) {
    using namespace mix;

    struct Case {
        int columns;
        bool graph = false;
    };

    for (Kind kind : {Kind::Block, Kind::Final}) {
        for (auto pattern : {mix::Pattern::Dense, mix::Pattern::Zero, mix::Pattern::InjectionOnly,
                             mix::Pattern::RowMapping, mix::Pattern::Saturation}) {
            if (kind == Kind::Final && pattern == mix::Pattern::InjectionOnly) continue;
            std::vector<Case> cases;
            if (pattern == mix::Pattern::Dense) {
                for (int m :
                     {1,    2,    3,    4,    5,    16,   31,   32,   33,  48,   49,   63,   64,
                      96,   97,   127,  128,  129,  159,  160,  161,  191, 192,  193,  223,  224,
                      225,  255,  256,  257,  319,  320,  321,  383,  384, 385,  447,  448,  449,
                      511,  512,  513,  671,  672,  673,  767,  768,  769, 1024, 1199, 1200, 1201,
                      1359, 1360, 1361, 1536, 1631, 1632, 1633, 1792, 2048})
                    cases.push_back({m});
                for (int m : {23, 24, 25, 575, 576, 577, 895, 896, 897, 1023, 1025, 1215, 1216,
                              1217, 2047, 2049, 4095, 4096, 4097})
                    cases.push_back({m});
                for (int m : {1, 2, 3, 4096}) cases.push_back({m, true});
                cases.push_back({56, true});
                cases.push_back({385, true});
                cases.push_back({0});
            } else {
                for (int m : {1, 3}) cases.push_back({m});
                if (pattern == mix::Pattern::RowMapping)
                    for (int m : {65, 385, 769}) cases.push_back({m});
            }
            std::vector<OpCase> catalogue;
            for (auto c : cases)
                catalogue.push_back(
                    {std::string(mix::name(kind)) + "/pattern" + std::to_string(int(pattern)) +
                         "/m" + std::to_string(c.columns) + (c.graph ? "/graph" : "/eager"),
                     {{"execution", c.graph ? "graph" : "eager"}}});
            suite.add(std::move(catalogue), [=](auto selected) {
                Weights weights(kind, pattern);
                const auto input   = input_catalog(pattern);
                const auto catalog = std::span{input}.first(
                    pattern == mix::Pattern::Dense ? input.size() : 3 * kTotal);
                const auto ref       = mix::reference(weights, catalog);
                const auto execution = cuda_environment().context.execution_view();
                int failures         = 0;
                for (auto i : selected) {
                    auto c = cases[i];
                    failures += c.columns
                                    ? run(weights, catalog, ref, c.columns, execution, c.graph)
                                    : mix::validate(weights, ref, execution);
                }
                return failures + weights.check();
            });
        }
    }
    suite.add("block-mix/serial-column-parity/m5", [] { return serial_column_parity(); });
}
} // namespace ninfer::test::hc
