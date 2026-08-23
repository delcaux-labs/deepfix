---
name: ux-flow-plan
description: Create UX-first plans as flow trees, then attach concrete function names, files, and implementation anchors after the high-level flow is clear.
user-invocable: true
---

# UX Flow Plan

Create a high-level architecture plan that starts from the user experience and only then maps to code-level anchors.

## Steps

1. Restate the feature goal in product terms, using the user's language.

2. Build the current-state UX flow as a tree:
   ```text
   User action
   └─ System behavior
      └─ Existing architectural layer
         └─ Relevant function/file anchor
   ```

3. Build the desired-state UX flow as a second tree:
   ```text
   User action
   └─ System behavior
      └─ New or changed architectural layer
         └─ Relevant function/file anchor
   ```

4. Identify the architectural boundary decisions:
   - Which layer detects the condition
   - Which layer owns side effects
   - Which layer updates UI/status
   - Which layer persists or mutates state

5. Attach implementation anchors after the trees, not before:
   - Function names
   - File paths
   - Existing abstractions to reuse
   - Tests that should cover the flow

6. End with a small decision list:
   - Recommended architecture
   - Alternatives rejected
   - Open questions or assumptions

## Rules

- Start with UX and system flow, not line numbers or code edits.
- Use tree diagrams for flows whenever possible.
- Keep function names and file paths attached as anchors, not as the main narrative.
- Do not over-specify implementation details before the architectural boundary is clear.
- Separate "current flow" from "desired flow".
- Call out whether a feature is coupled to an existing concept or intentionally independent from it.
- Prefer plain product and architecture language over framework-specific jargon unless the jargon names a real boundary.
