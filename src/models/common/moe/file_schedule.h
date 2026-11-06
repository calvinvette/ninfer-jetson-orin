#pragma once

#include "models/common/moe/schedule.h"

namespace ninfer::models::moe {

// GPU-only schedule for metadata-backed expert banks. It stages one complete bank at a time
// from file storage into a stable device cache before invoking the normal fused MoE Op.
[[nodiscard]] std::shared_ptr<const Plan>
describe_file_schedule(std::span<const ExpertBankCacheSpec> banks,
                       std::span<const LayerSpec> layers);

} // namespace ninfer::models::moe
