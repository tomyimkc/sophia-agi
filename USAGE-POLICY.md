# Acceptable Use Policy — Sophia AGI

> This policy governs **use of the Sophia models, weights, adapters, and the
> released corpus** ("the Artifacts"). The *source code* is licensed under
> Apache-2.0 (see `LICENSE`); this policy adds a **use-based restriction on the
> Artifacts** in the spirit of a Responsible-AI (RAIL) license. By downloading,
> running, fine-tuning, distributing, or otherwise using the Artifacts you agree
> to these terms. If you cannot comply, do not use the Artifacts.

The project is transparency-first and permissive by design. This policy exists for
one reason: so that **misuse is a violation of the terms you accepted**, not
merely something the author disapproves of. It does not — and cannot — technically
prevent a determined bad actor who forks the code, but it (a) makes intent
explicit, (b) gives the maintainer standing to revoke a grant and issue takedowns,
and (c) is the use-based term attached to the gated model repositories.

## 1. You must NOT use the Artifacts to

1. **Cause physical or severe harm** — including providing operational uplift for
   weapons (CBRN: chemical, biological, radiological, nuclear), explosives, or
   other instruments whose primary purpose is to injure or kill.
2. **Generate child sexual abuse material (CSAM)** or any sexual content involving
   minors, or **non-consensual intimate imagery (NCII)** of real persons.
3. **Conduct targeted harassment, stalking, doxxing, or intimidation**, or
   generate content intended to demean a person or group on the basis of a
   protected attribute.
4. **Commit fraud or deception** — phishing, scams, impersonation of real people
   or organizations, fake reviews, academic-integrity violations, or
   disinformation campaigns / coordinated inauthentic behavior.
5. **Develop malware or attack systems** — generate, improve, or operate
   ransomware, exploits, or intrusion tooling against systems you are not
   explicitly authorized to test.
6. **Run mass surveillance or social scoring** that violates human rights, or
   biometric identification of individuals without a lawful basis and consent.
7. **Provide unqualified high-stakes advice as if authoritative** — medical,
   legal, or financial decisions presented as professional advice without a
   qualified human in the loop and clear disclaimers.
8. **Violate law or sanctions** — including export-control / OFAC-sanctioned
   destinations and parties, or any applicable data-protection law.
9. **Remove, disable, or circumvent safety mitigations** of a *published* Sophia
   model (the refusal layer, output guard, or conscience gate) and then
   **redistribute** the resulting artifact, or present it as "Sophia." Research
   on robustness is permitted (see §3); redistribution of a de-safetied build to
   third parties is not.
10. **Misrepresent capability** — claim the Artifacts are proven AGI, or attach
    the project's marks to such claims (see `TRADEMARK-POLICY.md` and the
    no-overclaim gate in `SECURITY.md`).

## 2. You must

- Comply with all applicable laws and third-party rights.
- Keep the **safety mitigations intact** in any public deployment, or clearly
  disclose that you have modified them and that the result is not endorsed.
- Pass this policy **downstream**: anyone you distribute the Artifacts (or a
  derivative) to must receive these terms.
- Attribute per `NOTICE.md` and retain license headers.

## 3. Permitted: security & safety research

Adversarial testing, red-teaming, jailbreak research, and robustness evaluation
**of your own deployment or of Sophia** are explicitly permitted and encouraged —
this is how the model gets safer. Use the harness in `tools/redteam_runner.py`.
Report findings responsibly per `SECURITY.md`. Do **not** point offensive tooling
at third-party systems without their authorization.

## 4. High-risk domains

Medical, legal, and financial outputs are **decision-support, not professional
advice**. They must carry a disclaimer and a human-in-the-loop. The repo ships
domain faithfulness checks (`agent/medical_faithfulness.py`,
`agent/legal_faithfulness.py`) — use them; they do not replace a qualified human.

## 5. Enforcement

Violating §1 or §2 **terminates your rights to the Artifacts** under this policy
immediately. The maintainer may request removal of derivative Artifacts that
breach this policy and may report illegal use to the relevant authorities. This
policy may be updated; the version in the release you obtained governs that copy.

## 6. Reporting

Report misuse, vulnerabilities, or safety issues per `SECURITY.md` (coordinated,
private disclosure). For leaked secrets, rotate first, then report.

---

*This Acceptable Use Policy supplements but does not override the Apache-2.0
license on the source code. Where the two address different subject matter (code
vs. Artifacts), both apply.*
