import wandb

api = wandb.Api()
staged_artifact = api.artifact('wandb-registry-model/book-genre-classifier:latest')
staged_artifact.aliases.append('production')
staged_artifact.save()

print('Promoted to production')