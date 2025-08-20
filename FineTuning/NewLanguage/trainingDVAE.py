import compatibility

import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.trainer.dvae_dataset import DVAEDataset
from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig


def train_DVAE(dvae_pretrained, mel_norm_file, metadatas, language="mt", lr=5e-6, grad_clip_norm=0.5, num_epochs=5, batch_size=512):
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please run this script on a machine with a GPU.")

  dvae = DiscreteVAE(
    channels=80,
    normalization=None,
    positional_dims=1,
    num_tokens=1024,
    codebook_dim=512,
    hidden_dim=512,
    num_resnet_blocks=3,
    kernel_size=3,
    num_layers=2,
    use_transposed_convs=False,
  )
  dvae.load_state_dict(torch.load(dvae_pretrained), strict=False).cuda()
  torch_mel_spectrogram_dvae = TorchMelSpectrogram(mel_norm_file=mel_norm_file, sampling_rate=22050).cuda()

  opt = Adam(dvae.parameters(), lr = lr)

  DATASETS_CONFIG_LIST = []
  for metadata in metadatas:
    train_csv, eval_csv, language = metadata.split(",")
    print(train_csv, eval_csv, language)
    config_dataset = BaseDatasetConfig(
      formatter="coqui",
      dataset_name="ft_dataset",
      path=os.path.dirname(train_csv),
      meta_file_train=os.path.basename(train_csv),
      meta_file_val=os.path.basename(eval_csv),
      language=language,
    )
    DATASETS_CONFIG_LIST.append(config_dataset)

  train_samples, eval_samples = load_tts_samples(
    DATASETS_CONFIG_LIST,
    eval_split=True,
    eval_split_max_size=256,
    eval_split_size=0.01,
  )
  eval_dataset = DVAEDataset(eval_samples, 22050, True, max_wav_len=15*22050)
  train_dataset = DVAEDataset(train_samples, 22050, False, max_wav_len=15*22050)

  eval_data_loader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=eval_dataset.collate_fn,
    num_workers=0,
    pin_memory=False
  )
  train_data_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=train_dataset.collate_fn,
    num_workers=4,
    pin_memory=False
  )

  torch.set_grad_enabled(True)
  dvae.train()


  def to_cuda(x: torch.Tensor) -> torch.Tensor:
    if x is None:
      return None
    if torch.is_tensor(x):
      x = x.contiguous()
      if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x

  @torch.no_grad()
  def format_batch(batch):
    if isinstance(batch, dict):
      for k, v in batch.items():
        batch[k] = to_cuda(v)
    elif isinstance(batch, list):
      batch = [to_cuda(v) for v in batch]

    try:
      batch['mel'] = torch_mel_spectrogram_dvae(batch['wav']) # type: ignore
      # if the mel spectogram is not divisible by 4 then input.shape != output.shape 
      remainder = batch['mel'].shape[-1] % 4 # type: ignore
      if remainder:
        batch['mel'] = batch['mel'][:, :, :-remainder] # type: ignore
    except NotImplementedError:
      pass
    return batch

  best_loss = 1e6

  for i in range(num_epochs):
    dvae.train()
    for cur_step, batch in enumerate(train_data_loader):
      opt.zero_grad()
      batch = format_batch(batch)
      recon_loss, commitment_loss, out = dvae(batch['mel']) # type: ignore
      recon_loss = recon_loss.mean()
      total_loss = recon_loss + commitment_loss
      total_loss.backward()
      clip_grad_norm_(dvae.parameters(), grad_clip_norm)
      opt.step()
      print(f"epoch: {i}", print(f"step: {cur_step}"), f'loss - {total_loss.item()}', f'recon_loss - {recon_loss.item()}', f'commit_loss - {commitment_loss.item()}')
      torch.cuda.empty_cache()
    
    with torch.no_grad():
      dvae.eval()
      eval_loss = 0
      for cur_step, batch in enumerate(eval_data_loader):
        batch = format_batch(batch)
        recon_loss, commitment_loss, out = dvae(batch['mel']) # type: ignore
        recon_loss = recon_loss.mean()
        eval_loss += (recon_loss + commitment_loss).item()
      eval_loss = eval_loss/len(eval_data_loader)
      if eval_loss < best_loss:
        best_loss = eval_loss
        torch.save(dvae.state_dict(), dvae_pretrained)
      print(f"#######################################\nepoch: {i}\tEVAL loss: {eval_loss}\n#######################################")

  print(f'Checkpoint saved at {dvae_pretrained}')


if __name__ == "__main__":
  from parsers import create_train_DVAE_parser
  parser =  create_train_DVAE_parser()
  args = parser.parse_args()

  trainer_out_path = train_DVAE(
    dvae_pretrained=args.dvae_pretrained,
    mel_norm_file=args.mel_norm_file,
    language=args.language,
    metadatas=args.metadatas,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    lr=args.lr
  )