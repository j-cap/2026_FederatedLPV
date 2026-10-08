# Repository working rules

## Commit and push after every experiment

The user requires every completed experiment, including a negative or failed
scientific result, to be committed and pushed before starting the next experiment.
The request to perform an experiment includes authorization to commit and push
its work to the appropriate repository branch. Do not ask for that authorization
again. This rule supersedes older protocol wording that treats pushing as a
separate action.

- Commit the implementation, protocol/configuration, frozen learned models,
  assignments, provenance locks, result summaries, audits and report changes.
  Frozen inputs needed by later experiments must have a durable saved copy.
- Push the completed work and verify the remote branch points to the intended
  commit. Local commits alone do not complete the experiment handoff.
- For long experiments, commit and push the protocol before execution and save
  reproducible checkpoints during execution. Keep large regenerable trajectories
  out of git when appropriate, but preserve their reconstruction inputs and hashes.
- If pushing fails, try the connected GitHub tools when available. State any
  remaining synchronization failure explicitly and preserve a recovery archive.
- Do not force-push unrelated history or alter historical locked files merely
  to update workflow wording. Record the branch and remote commit in the final
  status. Distinguish recovered original artifacts from reconstructed experiments.
