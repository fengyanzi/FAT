import os
import torch
from torchvision import transforms
from PIL import Image
import argparse
from models.FAT import FAT
#from models.FASTformerv2 import MST_net as FAT
from tqdm import tqdm


def unnormalize(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'Using device: {device}')

    # 初始化模型
    model = FAT().to(device)
    print("Loading model: {}".format(args.model_path))
    model.load_state_dict(torch.load(args.model_path),  strict=False)

    # 启用FP16推理
    if args.fp16 and torch.cuda.is_available():
        model = model.half()
        print("Enabled FP16 inference")

    if args.test_img is None or not os.path.isdir(args.test_img):
        raise FileExistsError("Please input a valid directory path.")

    for filename in tqdm(os.listdir(args.test_img)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            cloud_img_path = os.path.join(args.test_img, filename)
            cloud_img = Image.open(cloud_img_path)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            with torch.no_grad():
                input_tensor = transform(cloud_img).unsqueeze(0).to(device)

                # 如果启用FP16，将输入转换为FP16
                if args.fp16 and torch.cuda.is_available():
                    input_tensor = input_tensor.half()

                out = model(input_tensor)
                out = unnormalize(out, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

                # 将输出转换回FP32以便保存
                if args.fp16 and torch.cuda.is_available():
                    out = out.float()

                out_image = transforms.ToPILImage()(out.clamp(0, 1).squeeze(0))

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            save_path = os.path.join(args.save_dir,
                                     os.path.splitext(filename)[0] + '.png')
            print(f"Saving result to: {save_path}...")
            out_image.save(save_path)
            print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--test_img', type=str, default=r'./datasets/test/',
                        help='Path to dataset')
    parser.add_argument('--model_path', type=str, default=r'./checkpoints/best.pth', help='Path to pretrained model')
    parser.add_argument('--save_dir', type=str, default=r'./result/version1/',
                        help='Path to save predict results')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--fp16', action='store_true', default=False, help='Enable FP16 inference')

    args = parser.parse_args()
    subtitle = """  


      ___           ___           ___           ___           ___                                           
     /\  \         /\  \         /\  \         /\  \         /\  \                                          
    /::\  \       /::\  \        \:\  \       /::\  \       /::\  \                                         
   /:/\:\  \     /:/\:\  \        \:\  \     /:/\ \  \     /:/\:\  \                                        
  /::\~\:\  \   /::\~\:\  \       /::\  \   _\:\~\ \  \   /::\~\:\  \                                       
 /:/\:\ \:\__\ /:/\:\ \:\__\     /:/\:\__\ /\ \:\ \ \__\ /:/\:\ \:\__\                                      
 \/__\:\ \/__/ \/__\:\/:/  /    /:/  \/__/ \:\ \:\ \/__/ \/_|::\/:/  /                                      
      \:\__\        \::/  /    /:/  /       \:\ \:\__\      |:|::/  /                                       
       \/__/        /:/  /     \/__/         \:\/:/  /      |:|\/__/                                        
                   /:/  /                     \::/  /       |:|  |                                          
                   \/__/                       \/__/         \|__|                                          
      ___           ___           ___           ___           ___       ___           ___           ___     
     /\  \         /\  \         /\  \         /\  \         /\__\     /\__\         /\  \         /\__\    
    /::\  \       /::\  \       /::\  \       /::\  \       /:/  /    /:/  /        /::\  \       /:/  /    
   /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/  /    /:/  /        /:/\:\  \     /:/__/     
  /:/  \:\  \   /:/  \:\  \   /:/  \:\  \   /:/  \:\__\   /:/  /    /:/  /  ___   /:/  \:\  \   /::\__\____ 
 /:/__/_\:\__\ /:/__/ \:\__\ /:/__/ \:\__\ /:/__/ \:|__| /:/__/    /:/__/  /\__\ /:/__/ \:\__\ /:/\:::::\__\ 
 \:\  /\ \/__/ \:\  \ /:/  / \:\  \ /:/  / \:\  \ /:/  / \:\  \    \:\  \ /:/  / \:\  \  \/__/ \/_|:|~~|~   
  \:\ \:\__\    \:\  /:/  /   \:\  /:/  /   \:\  /:/  /   \:\  \    \:\  /:/  /   \:\  \          |:|  |    
   \:\/:/  /     \:\/:/  /     \:\/:/  /     \:\/:/  /     \:\  \    \:\/:/  /     \:\  \         |:|  |    
    \::/  /       \::/  /       \::/  /       \::/__/       \:\__\    \::/  /       \:\__\        |:|  |    
     \/__/         \/__/         \/__/         ~~            \/__/     \/__/         \/__/         \|__|      
    """
    print(subtitle)

    main(args)