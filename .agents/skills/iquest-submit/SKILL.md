---
name: iquest-submit
description: >-
  Submit, run, and inspect CUDA C++, Triton, or CuTe DSL solutions on IQuest Competition
  (competition.iquestlab.com) via the built-in browser subagent. Use when the user asks to
  test, submit, check leaderboard/scores, or iterate on a problem at competition.iquestlab.com.
---

# IQuest Competition Browser Submission Workflow

Standard operating procedure for reliably inserting code into the Monaco editor, running sample checks, submitting to NVIDIA H20, and collecting per-case benchmark results on `https://competition.iquestlab.com`.

## Pre-Flight Rules

- **One active evaluation at a time**: The platform enforces one concurrent Run/Submit per account. Wait for the current evaluation to finish before triggering another.
- **Stage the source file before invoking `browser`**: The `browser` subagent is sandboxed and cannot read arbitrary workspace paths. Copy the code file to:
  ```
  <Artifact Directory Path>/browser_sandbox/references/<filename>
  ```
  and pass that full path in the prompt so the subagent can read it with `read_file`.
- **Reuse the same browser subagent** via `send_message` for follow-up runs/submits on the same problem page so the logged-in tab stays open.
- **Always `Run` before `Submit`**: `Run` compiles and verifies the sample test case in ~3–8 seconds without consuming one of the contest's `100` Submit attempts. Only click `Submit` after `Run` passes (or if the user explicitly asks to submit directly).

## Reliable Code Insertion into Monaco Editor

The built-in `webview_browser` sandbox does **not** expose `page.evaluate`, `window`, or `document`, and Monaco's hidden `<textarea>` reacts badly to naive Playwright inputs:

- **NEVER use `locator.fill(code)` on Monaco** — it doubles lines and corrupts indentation.
- **NEVER use `keyboard.type(code)`** — it is slow and triggers auto-closing braces/quotes.

### Verified Method for CUDA C++ (and other brace-delimited languages)

Use **`page.keyboard.insertText()`** after clearing the editor and stripping leading whitespace per line (prevents leading spaces from compounding on top of Monaco's auto-indent, avoiding bloated lines or exceeding the 524,288-char limit while preserving exact C++ semantics):

```js
await page.setViewportSize({ width: 1280, height: 800 });

// 1. Reset starter template cleanly (optional if editor already has clean state)
// Click the Reset code icon button and confirm "Reset code" in the dialog.

// 2. Focus Monaco and delete all existing lines
const editor = page.locator('.monaco-editor').first();
await editor.locator('.view-line').last().click();
await page.keyboard.press('Meta+a');   // macOS; use Control+a on Linux/Windows
await page.keyboard.press('Backspace');
await page.waitForTimeout(200);

// 3. Strip leading whitespace per line and insert atomically in ONE call
const stripped = rawCode
  .replace(/\r\n/g, '\n')
  .split('\n')
  .map(line => line.trimStart())
  .join('\n');
await page.keyboard.insertText(stripped);
await page.waitForTimeout(300);
```

### Verification Before Clicking `Run`

Before submitting, inspect `.monaco-editor .view-lines` bounding box / style:
- Height should be approximately `20px × expectedLineCount` (confirms no line doubling).
- Width should stay compact (e.g., `< 1500px`, confirms no cascading auto-indent).
- Confirm the language dropdown (`CUDA C++`, `Triton`, or `CuTe DSL`) and target GPU (`NVIDIA H20`) match the code.

### Note for Python / Triton / CuTe DSL

Python syntax is indentation-sensitive, so **do not** strip leading whitespace with `trimStart()`. Instead, test insertion on a short 5-line indented block first to check whether `keyboard.insertText()` compounds indentation in Python mode; if it does, disable auto-indent or paste via clipboard (`Meta+v`) rather than stripping leading spaces.

## Running, Submitting, and Collecting Results

1. **Sample Run (`Run`)**:
   - Click `page.getByRole('button', { name: 'Run' })`.
   - Wait until `Run` is re-enabled (`page.locator('button:not([disabled])').filter({ hasText: 'Run' }).waitFor({ state: 'visible', timeout: 60000 })`).
   - Read the `Sample run` tabpanel text. Confirm `Passed`. If there is a compiler or runtime error, stop and return the exact diagnostic without clicking `Submit`.
2. **Official Submit (`Submit`)**:
   - Click `page.getByRole('button', { name: 'Submit' })`.
   - Wait until evaluation finishes (~8–30s; up to 5 minutes limit).
   - Read the `Submission` tabpanel for:
     - Status (`Accepted` / `Wrong Answer` / `Compile Error`, `X / N` test cases passed)
     - Official `Contest points` and `Total GPU time`
     - Per-test-case runtime breakdown
   - Open `My submissions` (or the submission URL `/submissions/<id>`) to capture exact **Throughput (GFLOPS)**, timestamp, submission ID, and remaining `Submits X / 100`.
3. **Report Back**:
   - Return the submission URL, status, score, total GPU time, throughput (GFLOPS), per-case timing table, and submits used.
