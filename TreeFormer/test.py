import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from network import pvt_cls as TCN
import torch.nn.functional as F
from scipy.io import savemat 
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--batch-size', type=int, default=8, help='train batch size')
parser.add_argument('--crop-size', type=int, default=256, help='the crop size of the train image')
parser.add_argument('--model-path', type=str, default='datasets/best_model.pth', help='saved model path')
parser.add_argument('--data-path', type=str, default='datasets', help='dataset path')
parser.add_argument('--dataset', type=str, default='TC')


def resolve_device(device_arg: str) -> torch.device:
    os.environ['CUDA_VISIBLE_DEVICES'] = device_arg
    if not torch.cuda.is_available():
        print('CUDA is not available. Falling back to CPU.')
        return torch.device('cpu')

    arch_list = set(torch.cuda.get_arch_list())
    major, minor = torch.cuda.get_device_capability(0)
    sm = f'sm_{major}{minor}'
    if sm not in arch_list:
        print(
            f'CUDA arch {sm} is not supported by this PyTorch build ({sorted(arch_list)}). '
            'Falling back to CPU.'
        )
        return torch.device('cpu')

    return torch.device('cuda')


def test(args, isSave = True):
    device = resolve_device(args.device)

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {model_path}. '
            'Provide a valid path with --model-path.'
        )

    test_dir = os.path.join(data_path, 'test_data')
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f'Test dataset folder not found: {test_dir}. '
            'Provide a valid dataset root with --data-path.'
        )

    os.makedirs('predictions', exist_ok=True)

    dataset = crowd.Crowd_TC(os.path.join(data_path, 'test_data'), crop_size, 1, method='val')
    if len(dataset) == 0:
        raise RuntimeError(
            f'No test images found in {os.path.join(test_dir, "images")}. '
            'Expected .jpg files.'
        )

    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True)
        
    model = TCN.pvt_treeformer(pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    image_errs = []
    result = []
    R2_es = []
    R2_gt = []
    l=0;
    for inputs, count, name, imgauss in dataloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            crop_imgs, crop_masks = [], []
            b, c, h, w = inputs.size()
            rh, rw = args.crop_size, args.crop_size
            
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                    mask = torch.zeros([b, 1, h, w]).to(device)
                    mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                    crop_masks.append(mask)
            crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))
            
            crop_preds = []
            nz, bz = crop_imgs.size(0), args.batch_size
            for i in range(0, nz, bz):
                
                gs, gt = i, min(nz, i + bz)
                crop_pred, _ = model(crop_imgs[gs:gt])
                crop_pred = crop_pred[0]
                
                _, _, h1, w1 = crop_pred.size()                
                crop_pred = F.interpolate(crop_pred, size=(h1 * 4, w1 * 4), mode='bilinear', align_corners=True) / 16
                crop_preds.append(crop_pred)
            crop_preds = torch.cat(crop_preds, dim=0)
            #import pdb;pdb.set_trace()
            
            # splice them to the original size
            idx = 0
            pred_map = torch.zeros([b, 1, h, w]).to(device)
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                    idx += 1
            # for the overlapping area, compute average value
            mask = crop_masks.sum(dim=0).unsqueeze(0)
            outputs = pred_map / mask
            
            outputs = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)/4
            outputs = pred_map / mask
            
            img_err = count[0].item() - torch.sum(outputs).item()
            R2_gt.append(count[0].item())
            R2_es.append(torch.sum(outputs).item())
            
            print("Img name: ", name, "Error: ", img_err, "GT count: ", count[0].item(), "Model out: ", torch.sum(outputs).item())
            image_errs.append(img_err)
            result.append([name, count[0].item(), torch.sum(outputs).item(), img_err])
            
            savemat('predictions/'+name[0]+'.mat', {'estimation':np.squeeze(outputs.cpu().data.numpy()), 
            'image': np.squeeze(inputs.cpu().data.numpy()), 'gt': np.squeeze(imgauss.cpu().data.numpy())})
            l=l+1
            
            
    image_errs = np.array(image_errs)
        
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    R_2 = r2_score(R2_gt,R2_es)

    print('{}: mae {}, mse {}, R2 {}\n'.format(model_path, mae, mse,R_2))

    if isSave:
        with open("test.txt","w") as f:
            for i in range(len(result)):
                f.write(str(result[i]).replace('[','').replace(']','').replace(',', ' ')+"\n")
            f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    test(args, isSave= True)

