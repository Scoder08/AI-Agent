import httpx, re, pathlib

# ── PR diff fetcher ────────────────────────────────────────────────────
_PR_RE = re.compile(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)(?:/.*)?$")

ROOT = pathlib.Path(__file__).resolve().parents[1] 

async def fetch_pr_diff(url: str, token: str | None = None) -> str | None:
    m = _PR_RE.match(url.strip())
    if not m:
        return None
    owner, repo, num = m.groups()
    hdrs = {"Accept": "application/vnd.github.v3.diff"}
    if token:
        hdrs["Authorization"] = f"Bearer {token}"          # PAT
    api = f"https://api.github.com/repos/{owner}/{repo}/pulls/{num}"
    async with httpx.AsyncClient() as client:
        r = await client.get(api, headers=hdrs)
        return r.text if r.status_code == 200 else None

# ── diff slicer ────────────────────────────────────────────────────────
_HUNK   = re.compile(r"^@@")        # hunk header
_FILE   = re.compile(r"^\+\+\+ b/") # '+++ b/...'

def slice_diff(diff: str, context: int = 3) -> str:
    """
    Keep:
      • every '+++ b/<file>' header
      • every '@@ … @@' hunk header
      • each +/- line
      • up to `context` unchanged lines *after* a streak of +/- lines
    """
    out, keep, ctx = [], False, 0
    for ln in diff.splitlines():
        if _FILE.match(ln):                   # file header
            out.append(ln)
            keep = False                      # reset until next hunk
            continue
        if _HUNK.match(ln):
            out.append(ln)
            keep, ctx = True, context
            continue
        if keep:
            if ln.startswith(("+", "-")):
                out.append(ln)
                ctx = context
            elif ctx:
                out.append(ln)
                ctx -= 1
    return "\n".join(out)

# ── diff annotator ─────────────────────────────────────────────────────
HUNK_RE = re.compile(r"@@ -(\d+),?\d* \+(\d+),?.*@@")
FILE_RE = re.compile(r"^\+\+\+ b/(.+)")

def annotate_diff(diff: str) -> str:
    """
    Prefix every +/- line with  'path:lineno:'  so the LLM can reference it.
    """
    out, cur_file = [], None
    old_ln = new_ln = None
    for ln in diff.splitlines():
        if ln.startswith("+++ "):
            cur_file = FILE_RE.match(ln).group(1)
            out.append(ln)
            continue
        if ln.startswith("@@"):
            m = HUNK_RE.match(ln)
            old_ln, new_ln = int(m.group(1)), int(m.group(2))
            out.append(ln)
            continue
        if ln.startswith("+"):
            out.append(f"{cur_file}:{new_ln}:{ln}")
            new_ln += 1
        elif ln.startswith("-"):
            out.append(f"{cur_file}:{old_ln}:{ln}")
            old_ln += 1
        else:
            out.append(ln)
            old_ln += 1
            new_ln += 1
    return "\n".join(out)

# utils/pr_utils.py
import ast, textwrap
def functions_touched(diff: str, root=ROOT) -> str:
    """
    Return full source of every function/class whose lines were edited.
    """
    # 1) map filename -> set(edited_line_numbers)
    edits = {}
    for ln in diff.splitlines():
        if ln.startswith(("+++ ", "--- ")):
            cur = ln[6:] if ln.startswith("+++ b/") else cur
        elif ln.startswith(("+", "-")) and ":" in ln:          # annotated form
            file, lnno, _ = ln.split(":", 2)
            edits.setdefault(file, set()).add(int(lnno))

    blocks = []
    for file, touched in edits.items():
        src = pathlib.Path(root / file).read_text()
        tree = ast.parse(src)
        for node in [n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]:
            if any(node.lineno <= ln <= node.end_lineno for ln in touched):
                snippet = "\n".join(src.splitlines()[node.lineno-1: node.end_lineno])
                blocks.append(f"# {file}:{node.lineno}\n{textwrap.dedent(snippet)}")
    return "\n\n".join(blocks[:20])   # safety cap

