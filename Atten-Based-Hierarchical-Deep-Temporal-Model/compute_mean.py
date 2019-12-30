import math
import torch
from dataloader_phase1 import VolleyballDataset

IMG_SIZE = 224
BATCH_SIZE = 1
NUM_FRAMES = 10


def collate_fn(batch):
    person_pixels, person_actions, group_info = zip(*batch)

    return person_pixels[0], person_actions[0], group_info


if __name__ == "__main__":

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(VolleyballDataset(x),
                                       batch_size=BATCH_SIZE,
                                       shuffle=False,
                                       num_workers=0,
                                       collate_fn=collate_fn)
        for x in ['train', 'val']
    }

    mean = torch.zeros(3, )
    std = torch.zeros(3, )
    nb_samples = 0.
    rmean = 0.6378
    gmean = 0.4613
    bmean = 0.1868
    samples = 408010.0
    rstd = 0.
    gstd = 0.
    bstd = 0.

    for phase in ['train', 'val']:
        for inputs, labels, group_info in dataloaders_dict[phase]:
            # print(labels)
            # pixels = [inputs[i:i + STEP] for i in range(0, len(inputs), STEP)]
            # actions = [
            #     labels[i:i + STEP / 10] for i in range(0, len(labels), STEP / 10)
            # ]
            # print(actions)

            # for i in range(len(pixels)):
            #     pixel = np.stack(pixels[i], axis=0)
            #     action = np.array(actions[i])
            person_num = len(inputs)
            batch_samples = person_num * NUM_FRAMES
            print(group_info)

            for pidx in range(person_num):
                for fidx in range(NUM_FRAMES):
                    pixel = inputs[pidx][fidx]

                    pixel = pixel.view(3, -1)

                    rpixel = pixel[0]
                    gpixel = pixel[1]
                    bpixel = pixel[2]

                    for idx in range(rpixel.shape[0]):
                        rstd += (rpixel[idx].item() - rmean)**2
                        gstd += (gpixel[idx].item() - gmean)**2
                        bstd += (bpixel[idx].item() - bmean)**2

                    mean += pixel.mean(-1)
                    std += pixel.std(-1)
            nb_samples += batch_samples

    total_pixel = nb_samples * IMG_SIZE * IMG_SIZE

    mean /= nb_samples
    std /= nb_samples
    raw_std = (rstd, gstd, bstd)
    new_std = (math.sqrt(rstd / total_pixel), math.sqrt(gstd / total_pixel),
               math.sqrt(bstd / total_pixel))

    print(mean, std, nb_samples, raw_std, new_std)

    with open('mean.txt', 'w+') as f:
        f.write(
            'mean:{}\nstd:{}\nnb_samples:{}\nraw_std:{}\nnew_std:{}\n'.format(
                mean, std, nb_samples, raw_std, new_std))
