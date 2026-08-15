# Issue tracker

GitHub Issues in `moritzbrantner/visual-analysis` are the durable queue. Slices
use canonical `parent`, `blocked_by`, and `scope` YAML frontmatter. Agent Loop
state uses `ready-for-agent`, `agent-loop:active`, `agent-loop:blocked`, and
`agent-loop:ready-to-merge`.
