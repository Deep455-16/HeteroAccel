// src/primitives/cpu/CpuBackendExecutor.h
// Phase 16: Concrete CPU backend executor.
//
// Wraps all CpuOps functions behind the IBackendExecutor interface.
// Always available — no device capability check required.
// Thread-safety: stateless; see IBackendExecutor docs.

#pragma once

#include "primitives/IBackendExecutor.h"
#include "primitives/cpu/CpuOps.h"

namespace agr {
namespace primitives {

class CpuBackendExecutor : public IBackendExecutor {
public:
    std::string name()        const override { return "CpuBackendExecutor"; }
    bool        isAvailable() const override { return true; }

    BackendCapabilities capabilities() const override {
        BackendCapabilities c;
        c.backendName           = "CpuBackendExecutor";
        c.available             = true;
        c.copy                  = true;
        c.fill                  = true;
        c.add                   = true;
        c.multiply              = true;
        c.scale                 = true;
        c.sum                   = true;
        c.maxVal                = true;
        c.deviceBufferAllocation = false; // CPU host buffers only
        c.hostToDevice          = false;
        c.deviceToHost          = false;
        return c;
    }

    BackendOpResult copy    (Tensor& dst, const Tensor& src)                  override { return cpu::copy(dst, src);           }
    BackendOpResult fill    (Tensor& dst, float value)                         override { return cpu::fill(dst, value);          }
    BackendOpResult add     (Tensor& dst, const Tensor& a, const Tensor& b)   override { return cpu::add(dst, a, b);            }
    BackendOpResult multiply(Tensor& dst, const Tensor& a, const Tensor& b)   override { return cpu::multiply(dst, a, b);       }
    BackendOpResult scale   (Tensor& dst, const Tensor& src, float scalar)    override { return cpu::scale(dst, src, scalar);   }
    BackendOpResult sum     (const Tensor& src, double* out_sum)               override { return cpu::sum(src, out_sum);         }
    BackendOpResult maxVal  (const Tensor& src, float* out_max)               override { return cpu::maxVal(src, out_max);      }
};

} // namespace primitives
} // namespace agr
