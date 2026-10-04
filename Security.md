# Security Policy

## Overview

HeteroAccel is a local, hardware-adaptive runtime for AI inference. It is designed to operate primarily on the user's local machine and interact with local inference engines, models, memory, storage, and hardware accelerators.

Security is an important part of the project because HeteroAccel may operate with:

* Local AI models
* User prompts and generated content
* Local files and model paths
* System memory and storage
* CPU and GPU resources
* Hardware and runtime configuration
* Local inference processes

This document explains how security vulnerabilities should be reported and the security practices currently followed by the project.

---

## Supported Versions

Security fixes are primarily applied to the latest development version of HeteroAccel.

| Version               | Supported      |
| --------------------- | -------------- |
| Latest `main`         | ✅ Yes          |
| Older releases        | ⚠️ Best effort |
| Unmaintained versions | ❌ No           |

Users are encouraged to keep HeteroAccel updated to the latest available version.

---

## Reporting a Vulnerability

If you discover a security vulnerability in HeteroAccel, please **do not disclose the vulnerability publicly through a GitHub issue, pull request, discussion, or public forum before it has been investigated**.

Instead, report the vulnerability privately through the security contact or private vulnerability-reporting mechanism associated with the repository.

When reporting a vulnerability, please provide as much of the following information as possible:

* A clear description of the vulnerability
* The affected HeteroAccel version or commit
* The operating system and architecture
* Hardware configuration, if relevant
* Steps required to reproduce the issue
* A minimal proof of concept, if available
* Expected behavior
* Actual behavior
* Potential security impact
* Any proposed mitigation or fix

Please avoid including real credentials, API keys, private documents, personal information, or other sensitive data in the report.

---

## What Should Be Reported?

Security issues may include, but are not limited to:

### Code Execution

* Unexpected execution of arbitrary code
* Unsafe command execution
* Malicious model or input leading to unintended code execution
* Memory corruption resulting in code execution

### Memory Safety

* Buffer overflows
* Use-after-free
* Out-of-bounds access
* Double-free conditions
* Invalid pointer usage
* Integer overflow/underflow that can affect memory safety

### File and Path Security

* Path traversal
* Unauthorized file access
* Unsafe handling of model paths
* Arbitrary file overwrite or deletion
* Unsafe temporary-file handling

### Resource Exhaustion

* Uncontrolled memory allocation
* CPU exhaustion
* GPU resource exhaustion
* Disk-space exhaustion
* Denial-of-service conditions caused by malicious workloads or inputs

### Process and Runtime Isolation

* Unauthorized interaction with other processes
* Privilege escalation
* Unsafe subprocess handling
* Unexpected access to system resources

### Dependency Vulnerabilities

Security vulnerabilities originating from dependencies used by HeteroAccel may also be reported.

This includes vulnerabilities affecting components such as:

* llama.cpp
* Vulkan-related components
* CUDA-related components
* CMake/build dependencies
* Other third-party runtime libraries

---

## Model and Input Security

HeteroAccel executes local AI workloads and therefore should not automatically be considered safe simply because a model is stored locally.

Users should obtain models from trusted sources.

A malicious or intentionally modified model may potentially exploit vulnerabilities in the inference engine or underlying dependencies.

HeteroAccel does not currently claim to provide comprehensive security validation or sandboxing of arbitrary model files.

Users should therefore treat untrusted models as potentially unsafe input.

---

## Local Data and Privacy

HeteroAccel is designed for local execution.

Under normal local execution, prompts, models, and inference data are intended to remain on the user's machine rather than being sent to a remote HeteroAccel service.

However, privacy depends on the complete application and inference stack being used.

For example:

```text
Application
     ↓
Inference Engine
     ↓
HeteroAccel
     ↓
Local Hardware
```

If the application itself communicates with external services, HeteroAccel cannot guarantee that those external communications are private.

HeteroAccel should therefore be considered a **local runtime component**, not a guarantee that every application using it is completely offline.

---

## Credentials and Secrets

Secrets should not be committed to the repository.

This includes:

* API keys
* Access tokens
* Passwords
* Private keys
* Cloud credentials
* Authentication tokens
* Personal credentials

Developers should use environment variables or an appropriate local secret-management mechanism where credentials are required by an integrating application.

If a secret is accidentally committed, it should be revoked or rotated immediately.

---

## Build and Dependency Security

HeteroAccel relies on third-party components and platform runtimes.

Developers should:

* Keep dependencies reasonably up to date
* Use trusted package and dependency sources
* Review dependency security advisories
* Avoid committing generated binaries unnecessarily
* Avoid committing credentials or private configuration
* Build from trusted source code
* Review changes to security-sensitive dependencies

Particular attention should be given to changes involving:

* llama.cpp
* memory management
* filesystem operations
* process execution
* backend loading
* Vulkan/CUDA integration
* model loading
* serialization/deserialization

---

## Runtime Security Principles

HeteroAccel follows several security-oriented design principles.

### Least Privilege

HeteroAccel should operate with the minimum operating-system privileges required by the application.

It should not require administrator/root privileges for normal local inference operation unless a specific platform integration requires them.

### Explicit Resource Management

Memory, device resources, model resources, and workload state should be managed through explicit runtime abstractions rather than uncontrolled global state.

### Input Validation

Runtime configuration and externally supplied parameters should be validated before being used.

This includes values related to:

* Context size
* Batch size
* Token limits
* GPU-layer configuration
* Thread configuration
* Model paths
* Resource sizes

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

### Cancellation Safety

Workload cancellation should terminate execution through controlled runtime mechanisms rather than abruptly terminating the entire process.

---

## Hardware Backend Security

HeteroAccel can interact with different hardware backends, including CPU, Vulkan, and CUDA where available.

Security properties may therefore depend on:

* Operating-system security
* GPU drivers
* Vulkan runtime/loader
* CUDA drivers/toolkit
* CPU architecture
* Backend implementation
* Inference engine implementation

HeteroAccel does not replace the security mechanisms provided by these underlying platforms.

Users should keep their operating system and hardware drivers reasonably up to date.

---

## Third-Party Dependencies

HeteroAccel integrates with external projects such as `llama.cpp` and hardware-specific runtime components.

A vulnerability in a third-party dependency may affect HeteroAccel even when the HeteroAccel code itself is not directly responsible for the vulnerability.

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

HeteroAccel is currently a developer-stage runtime and should not be considered a security boundary or sandbox.

In particular, HeteroAccel does not currently guarantee:

* Complete sandboxing of models
* Complete isolation from the host operating system
* Protection against malicious inference-engine code
* Protection against compromised GPU drivers
* Protection against compromised dependencies
* Formal verification of memory safety
* Security isolation between arbitrary workloads
* Protection against every possible denial-of-service condition

Applications requiring strong isolation should use appropriate operating-system, container, virtual-machine, or sandboxing mechanisms in addition to HeteroAccel.

---

## Security Best Practices for Users

Users should:

* Keep HeteroAccel updated
* Use trusted model sources
* Keep operating-system updates enabled
* Keep GPU drivers updated
* Avoid running untrusted binaries
* Avoid running HeteroAccel with unnecessary administrator privileges
* Protect local model and application files
* Never commit credentials to the repository
* Review third-party dependencies when building from source

For sensitive workloads, users should also consider operating-system-level isolation and filesystem permissions.

---

## Security Best Practices for Contributors

Contributors should:

* Validate externally supplied input
* Avoid unsafe memory operations
* Avoid unnecessary elevated privileges
* Avoid hard-coded credentials
* Avoid committing secrets
* Use safe filesystem APIs
* Handle errors explicitly
* Prefer RAII and ownership-safe C++ patterns
* Review changes affecting memory, process, filesystem, and backend operations
* Add regression tests for security-sensitive fixes

Security-sensitive changes should receive additional review before being merged.

---

## Scope

This security policy applies to the HeteroAccel project and its directly maintained runtime code.

Security issues belonging exclusively to an external dependency should also be reported to the relevant upstream project when appropriate.

Examples include:

* Operating-system vulnerabilities
* GPU driver vulnerabilities
* Vulkan loader vulnerabilities
* CUDA driver/toolkit vulnerabilities
* llama.cpp vulnerabilities that are independent of HeteroAccel changes

However, if the vulnerability is exposed or amplified specifically by HeteroAccel's integration with such a dependency, please report it to the HeteroAccel maintainers as well.

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
