# Security Policy

## Overview

HeteroAccel is a local, hardware-adaptive runtime for AI inference. Through Phase 22, its architecture includes hardware and backend abstractions, resource management and scheduling, model inspection and execution planning, model residency and streaming, a heterogeneous execution graph, native execution primitives, the HydroXL native inference-engine subsystem (CPU and Vulkan compute paths), a hardware-aware engine orchestration layer, and a native-DLL plugin loader.

It is designed primarily for local execution and may interact with local inference engines, models, memory, storage, and hardware accelerators. Security properties depend on the implemented code, enabled backends, dependencies, and the host environment.

Security is important because HeteroAccel may operate with:

- Local AI models
- User prompts and generated content
- Local files and model paths
- System memory and storage
- CPU and GPU resources
- Hardware and runtime configuration
- Local inference processes
- Model inspection and execution planning
- Model residency, streaming, prefetching, and eviction
- Heterogeneous execution graphs and native tensor/buffer operations
- HydroXL native inference-engine code, dequantization routines, and Vulkan compute shaders
- Engine orchestration and backend selection
- Native plugin DLLs loaded into the host process

This document explains how security vulnerabilities should be reported and describes security-oriented design expectations for the project. It is not a certification that every listed mitigation is implemented or that the runtime has undergone a formal security audit.

---

## Supported Versions

Security fixes are primarily expected to be developed against the latest maintained version of HeteroAccel. Actual support status depends on the maintainers and the project's release practices.

| Version | Supported |
|---|---|
| Latest `main` | Yes |
| Older releases | Best effort |
| Unmaintained versions | No |

Users are encouraged to keep HeteroAccel updated to the latest available version.

---

## Reporting a Vulnerability

If you discover a security vulnerability in HeteroAccel, **do not disclose the vulnerability publicly through a GitHub issue, pull request, discussion, or public forum before it has been investigated.**

Use GitHub's private vulnerability reporting feature for the repository if it is enabled, or contact the maintainers through a verified private security contact. Do not assume a private reporting feature is enabled unless it is visible on the repository. If no private channel is available, request a secure reporting route without posting exploit details publicly.

When reporting a vulnerability, please provide as much of the following information as possible:

- A clear description of the vulnerability
- The affected HeteroAccel version or commit
- The operating system and architecture
- Hardware configuration, including GPU model and driver version, if relevant
- The backend or engine involved (for example llama.cpp, HydroXL CPU, HydroXL Vulkan, or a plugin)
- Steps required to reproduce the issue
- A minimal proof of concept, if available
- Expected behavior
- Actual behavior
- Potential security impact
- Any proposed mitigation or fix

Please avoid including real credentials, API keys, private documents, personal information, or other sensitive data in the report.

---

## What Should Be Reported?

Security issues may include, but are not limited to:

### Code Execution

- Unexpected execution of arbitrary code
- Unsafe command execution
- Malicious model or input leading to unintended code execution
- Memory corruption resulting in code execution
- Plugin discovery or loading executing code from an unintended location

### Memory Safety

- Buffer overflows
- Use-after-free
- Out-of-bounds access
- Double-free conditions
- Invalid pointer usage
- Integer overflow or underflow affecting memory safety
- Out-of-bounds access in GGUF parsing, dequantization, tensor, or KV-cache code

### File and Path Security

- Path traversal
- Unauthorized file access
- Unsafe handling of model paths
- Unsafe handling of plugin or shader search paths
- Arbitrary file overwrite or deletion
- Unsafe temporary-file handling

### Resource Exhaustion

- Uncontrolled memory allocation
- CPU exhaustion
- GPU resource exhaustion
- Disk-space exhaustion
- Denial-of-service conditions caused by malicious workloads or inputs

### Process and Runtime Isolation

- Unauthorized interaction with other processes
- Privilege escalation
- Unsafe subprocess handling
- Unexpected access to system resources

### Dependency Vulnerabilities

Security vulnerabilities originating from dependencies used by HeteroAccel may also be reported.

This includes vulnerabilities affecting components such as:

- `llama.cpp` (fetched at configure time)
- Vulkan-related components (loader, SDK, shader compiler, GPU drivers)
- CUDA-related components (driver library probed at runtime)
- CMake and build dependencies
- Other third-party runtime libraries

---

## Model and Input Security

HeteroAccel can execute local AI workloads through engine integrations and native HydroXL execution paths. A model must not be considered safe merely because it is stored locally or uses a supported file format.

Users should obtain models from trusted sources. A malicious or intentionally modified model may potentially exploit vulnerabilities in the inference engine or underlying dependencies.

Models are parsed by more than one component: llama.cpp (for the llama.cpp execution path and GGUF inspection) and HydroXL's own GGUF loader (for the native path). HydroXL's loader and model code include checks for tensor dimensionality, 64-bit size overflow, data offsets against file size, minimum raw-byte sizes per quantization format, and context limits. These checks reduce risk but are not a substitute for independent review, and no fuzzing or formal audit of the parsers is documented.

HeteroAccel does not claim to provide comprehensive security validation, memory-safe parsing, or sandboxing of arbitrary model files. Treat untrusted model files as potentially hostile input and use operating-system-level isolation where appropriate.

---

## Local Data and Privacy

HeteroAccel is intended to support local execution and is not inherently a remote inference service. Whether prompts, models, telemetry, or other data leave the machine depends on the application, selected engine, optional integrations, and any external services they use.

Verify the behavior of the complete stack rather than assuming that local runtime use guarantees offline operation.

A typical local execution architecture may look like this:

```text
Application
     ↓
Inference Engine
     ↓
HeteroAccel
     ↓
Local Hardware
```

If the application or an integrated engine communicates with external services, HeteroAccel cannot guarantee that those external communications are private.

Note that the **build** is not offline: the first CMake configure downloads llama.cpp from GitHub. Runtime profiling and performance data may also be stored locally (see the `profile` command); protect those files as you would other local application data.

HeteroAccel should therefore be considered a **local runtime component**, not a guarantee that every application using it is completely offline.

---

## Credentials and Secrets

Secrets should not be committed to the repository.

This includes:

- API keys
- Access tokens
- Passwords
- Private keys
- Cloud credentials
- Authentication tokens
- Personal credentials

Developers should use environment variables or an appropriate local secret-management mechanism where credentials are required by an integrating application.

If a secret is accidentally committed, it should be revoked or rotated immediately.

---

## Build and Dependency Security

HeteroAccel relies on third-party components and platform runtimes.

Developers should:

- Keep dependencies reasonably up to date
- Use trusted package and dependency sources
- Review dependency security advisories
- Avoid committing generated binaries (compiled SPIR-V and model files are git-ignored)
- Avoid committing credentials or private configuration
- Build from trusted source code
- Review changes to security-sensitive dependencies

The build fetches llama.cpp with CMake `FetchContent`, pinned to a release **tag** rather than a commit hash. A tag can in principle be moved upstream; consider pinning to a commit hash for higher assurance. Vulkan compute shaders are compiled from `shaders/*.comp` to SPIR-V at build time with `glslangValidator` or `glslc` from the Vulkan SDK, so the SDK and its tools are part of the build trust boundary.

Particular attention should be given to changes involving:

- `llama.cpp` and other inference-engine code
- HydroXL tensor operations, dequantization, and model parsing
- Vulkan compute shaders and the code that loads and dispatches them
- Memory management and tensor/buffer lifetimes
- Model residency, streaming, prefetching, and eviction
- Filesystem operations and model paths
- Process execution and plugin loading
- Backend selection, orchestration, and Vulkan/CUDA integration
- Execution-graph scheduling, cancellation, and error handling
- Serialization and deserialization

---

## Runtime Security Principles

HeteroAccel follows several security-oriented design principles.

### Least Privilege

HeteroAccel should operate with the minimum operating-system privileges required by the application.

It should not require administrator or root privileges for normal local inference operation unless a specific platform integration requires them.

### Explicit Resource Management

Memory, device resources, model resources, and workload state should be managed through explicit runtime abstractions rather than uncontrolled global state.

### Input Validation

Runtime configuration and externally supplied parameters should be validated before being used.

This includes values related to:

- Context size
- Batch size
- Token limits
- GPU-layer configuration
- Thread configuration
- Model paths
- Plugin and shader paths
- Resource sizes

Configuration can also arrive through environment variables, notably `HETEROACCEL_MODEL_PATH` (model location) and `AGR_PLUGIN_DIR` (plugin search directory). Treat these as trusted-only inputs: whoever controls the process environment controls what is loaded.

### Safe Failure

When a workload cannot safely execute under the current resource constraints, the runtime should prefer controlled failure or fallback behavior rather than continuing with an unsafe configuration.

For example:

```text
Memory Pressure
       ↓
Execution Policy
       ↓
Restricted / CPU Fallback
```

The orchestration layer follows the same idea: if a selected backend (for example the HydroXL Vulkan engine) fails to initialize, it can fall back to the HydroXL CPU engine and report that a fallback occurred.

### Cancellation Safety

Workload cancellation should terminate execution through controlled runtime mechanisms rather than abruptly terminating the entire process.

### Model Residency, Streaming, and Native Execution

Model-region extraction, file-backed reads, prefetching, eviction, tensor allocation, and native dequantization are security-sensitive operations.

Implementations should:

- Validate file offsets and data sizes.
- Check integer arithmetic for overflow.
- Handle short reads and I/O errors.
- Enforce appropriate resource limits.
- Maintain clear ownership and lifetime rules.

These are security requirements to verify in code and tests, not assumptions that every path is already hardened.

### Execution Graphs and Engine Boundaries

Execution-graph dependencies, cancellation, backend transfers, and engine lifecycle transitions should be validated.

A common engine interface, registry, or orchestrator does not provide process isolation.

### Plugins

The plugin loader (Windows DLLs, `src/plugin`) loads plugins with `LoadLibrary` **into the host process with full host-process privileges and no sandbox**. Important properties:

- Discovery loads each candidate DLL temporarily to check its ABI and identity, so **plugin code (including `DllMain`) can execute during discovery**, not only when a plugin is explicitly run.
- The plugin directory is `./plugins` beside the executable by default and can be changed with the `AGR_PLUGIN_DIR` environment variable.
- The ABI version check protects against version mismatch, not against malicious plugins. It is not a trust or signature mechanism, and no signature verification is implemented.
- Plugins are not currently registered as execution engines and are not connected to the orchestrator; they are invoked through the loader's execute call.

Do not load or execute untrusted native plugins. Restrict write access to the plugin directory and to the executable's directory, and do not point `AGR_PLUGIN_DIR` at locations writable by other users.

---

## Hardware Backend Security

HeteroAccel's architecture includes backend abstractions for CPU and supported accelerator paths, including Vulkan where implemented. CUDA is detected by probing the NVIDIA driver library at runtime, but CUDA execution is not enabled in the current build.

Detection of a backend does not establish that every operation, model, or execution-graph node is supported by it.

Security properties may depend on:

- Operating-system security
- GPU drivers
- Vulkan runtime and loader
- CUDA drivers and toolkit
- CPU architecture
- Backend implementation
- Inference-engine implementation

HeteroAccel does not replace the security mechanisms provided by these underlying platforms. GPU memory isolation between processes, and the handling of malformed compute shaders or buffers, is provided by the driver and operating system, not by HeteroAccel.

The HydroXL Vulkan path loads compiled SPIR-V shader files from a `shaders` directory beside the executable (a build-tree fallback path is compiled in for development). Protect these files and the installation directory against modification: altered shaders would be executed on the GPU. The CUDA detector loads the driver library by name at runtime; keep the system and application directories free of untrusted DLLs.

Users should keep their operating system and hardware drivers reasonably up to date.

---

## Third-Party Dependencies

HeteroAccel has included integration with `llama.cpp` and hardware-specific runtime components. HydroXL is the native inference-engine subsystem introduced in Phase 17 and extended with a Vulkan compute path in Phase 22.

The plugin loader exists but is not wired into the engine registry or orchestrator. Optional external-engine adapters beyond llama.cpp are separate roadmap items unless verified as implemented in the current checkout.

A vulnerability in a third-party dependency may affect HeteroAccel even when the HeteroAccel code itself is not directly responsible.

When a relevant upstream vulnerability is identified, the project should evaluate:

1. Whether HeteroAccel is affected
2. Which versions are affected
3. Whether an upstream update is available
4. Whether a local mitigation is possible
5. Whether users need to upgrade

---

## Security Disclosure Process

After receiving a private vulnerability report, the maintainers will attempt to:

1. Acknowledge the report
2. Reproduce and validate the vulnerability
3. Determine its security impact
4. Identify affected versions
5. Develop or coordinate a fix
6. Test the fix
7. Release the appropriate update
8. Provide disclosure information when appropriate

The exact response time may depend on the severity and complexity of the vulnerability.

---

## Responsible Disclosure

Please allow the maintainers reasonable time to investigate and address a vulnerability before publicly disclosing detailed exploit information.

Coordinated disclosure helps protect users who have not yet had an opportunity to update.

Researchers who responsibly report valid vulnerabilities are appreciated and will be credited when appropriate and when they wish to be identified.

---

## Security Limitations

HeteroAccel remains a developing runtime, including its native execution, HydroXL, orchestration, and plugin components, and should not be considered a security boundary or sandbox.

The architecture's abstractions and roadmap do not by themselves establish that all security controls have been implemented, audited, or tested.

In particular, HeteroAccel does not currently guarantee:

- Complete sandboxing or security validation of model files
- Complete isolation from the host operating system
- Protection against malicious inference-engine or native HydroXL code
- Sandboxing, signing, or trust verification of plugins
- A verified security boundary between execution-graph nodes or workloads
- Protection against compromised GPU drivers
- Protection against compromised dependencies or build tools
- Formal verification of memory safety
- Fuzz-tested or audited model and GGUF parsers
- Security isolation between arbitrary workloads
- Protection against every possible denial-of-service condition

Applications requiring strong isolation should use appropriate operating-system, container, virtual-machine, or sandboxing mechanisms in addition to HeteroAccel.

---

## Security Best Practices for Users

Users should:

- Keep HeteroAccel updated
- Use trusted model sources
- Load only plugins you built or trust, from a directory only you can write to
- Keep operating-system updates enabled
- Keep GPU drivers updated
- Avoid running untrusted binaries
- Avoid running HeteroAccel with unnecessary administrator privileges
- Protect local model, shader, plugin, and application files
- Never commit credentials to the repository
- Review third-party dependencies when building from source

For sensitive workloads, users should also consider operating-system-level isolation and filesystem permissions.

---

## Security Best Practices for Contributors

Contributors should:

- Validate externally supplied input
- Avoid unsafe memory operations
- Avoid unnecessary elevated privileges
- Avoid hard-coded credentials
- Avoid committing secrets
- Use safe filesystem APIs
- Handle errors explicitly
- Prefer RAII and ownership-safe C++ patterns
- Review changes affecting memory, process, filesystem, plugin-loading, and backend operations
- Add regression tests for security-sensitive fixes

Security-sensitive changes should receive additional review before being merged.

---

## Scope

This security policy applies to the HeteroAccel project and its directly maintained runtime code, including model inspection and planning, residency and streaming, execution-graph infrastructure, native execution primitives, HydroXL code and Vulkan compute shaders, the orchestration layer, and the plugin loader.

The inclusion of a component in this scope does not imply that it has received a formal security audit.

Security issues belonging exclusively to an external dependency should also be reported to the relevant upstream project when appropriate.

Examples include:

- Operating-system vulnerabilities
- GPU driver vulnerabilities
- Vulkan loader vulnerabilities
- CUDA driver/toolkit vulnerabilities
- `llama.cpp` vulnerabilities that are independent of HeteroAccel changes

However, if a vulnerability is exposed or amplified specifically by HeteroAccel's integration with such a dependency, please report it to the HeteroAccel maintainers as well.

Third-party plugins are outside this policy's scope; the plugin loader itself is in scope.

---

## Contact

For security issues, use the repository's **private security reporting mechanism** rather than creating a public GitHub issue.

Do not publicly disclose sensitive vulnerability details before the maintainers have had a reasonable opportunity to investigate and address the issue.

---

## Security Philosophy

HeteroAccel's security philosophy is based on a simple principle:

> **Adaptive hardware execution should not come at the cost of uncontrolled access to the user's system or data.**

The runtime should therefore prioritize:

```text
Validated Input
      ↓
Controlled Resource Access
      ↓
Explicit Runtime Policies
      ↓
Safe Execution
      ↓
Controlled Failure / Fallback
```

HeteroAccel is designed to remain a local execution and resource-management layer while leaving application-level security, operating-system security, and inference-engine security responsibilities with their respective components.
