// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"

namespace onnxruntime {

class VulkanFence {
 public:
  explicit VulkanFence(const VkDevice& logical_device);

  virtual ~VulkanFence();

  VkFence Get() const {
    return fence_;
  }

  VkResult Reset() const;

  VkResult Wait() const;

  bool SupportFenceFd() const;

 private:
  VkResult RawWait() const;

 private:
  const VkDevice& logical_device_;
  VkFence fence_;
};

}  // namespace onnxruntime