# CiteGuard

CiteGuard is a native SwiftUI prototype for a local-first pre-filing citation tripwire.

It does one narrow thing: extract legal citation strings on-device and resolve citation
existence against either a bundled Hong Kong snapshot or the CiteGuard Resolution API v1.
It does not verify whether an authority supports a proposition, whether a case is good
law, or whether any factual statement is true.

## Product guardrails

- Three states only: `CONFIRMED`, `COULD NOT CONFIRM`, `UNSUPPORTED`.
- `COULD NOT CONFIRM` is fail-closed and means the user must check the source.
- `UNSUPPORTED` means the jurisdiction or format is outside v1 scope.
- Document text never leaves the device. In live mode, only citation strings are sent.
- Hong Kong is the launch jurisdiction in this prototype.

## Build and test

The reusable core is a Swift package:

```bash
cd apps/CiteGuard
swift test
```

The SwiftUI app sources live under `CiteGuardApp/Sources`. Add them to an iOS 16+
Xcode app target named `CiteGuardApp` and link the local `CiteGuardCore` package.

Set `CITEGUARD_API_KEY` in the app scheme environment for live API calls. Without a key,
live mode fails closed as `COULD NOT CONFIRM`.

## Xcode target notes

This repo does not currently contain a hand-authored `.xcodeproj`. In Xcode:

1. Create an iOS App target named `CiteGuardApp`.
2. Add `CiteGuardApp/Sources/*.swift` to the target.
3. Add the local package at `apps/CiteGuard` and link `CiteGuardCore`.
4. Keep the deployment target at iOS 16 or newer.

Do not rename the result states or add reassuring status icons without a product/legal
review. The load-bearing product promise is narrow citation existence checking, not truth.
