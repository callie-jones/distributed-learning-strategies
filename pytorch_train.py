from torch.distributed.elastic.multiprocessing.errors import record

def train():
  for batch in iter(dataset):
    train_step(batch)

    if should_checkpoint:
      save_checkpoint(checkpoint_path)

@record
def main():
  load_checkpoint(checkpoint_path)
  initialize()
  train()

if __name__ == "__main__":
    main()