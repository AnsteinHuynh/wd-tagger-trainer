Hewwo~! 🐾(˶˃ ᵕ ˂˶) .ᐟ.ᐟ⊹˖ᯓ★. ݁₊

If your model is image capable begin by reading C:/2071-sophie.png
Your scratch space directory is $WS/scratch/sophie/ where $WS is project workspace root, not a literal string
Freely summon assistant(s) if needed


I want to train a good anime tagger! We're going to use DinoV3 as a base and going to train the OFTv2 (Orthogonal Fine Tune) adapter for it, using optimizer SinkSGD_ADV (which is muon-like)

I don't want to wait around too much, so I want to train only a maximum of 10,000 pictures in one epoch.

I want maximum training efficiency. That means BF16 is used where appropriate.


Ask me questions if needed!

# Links:
Model:
[DinoV3-ViTB](I:\Models\dinov3-vitb16-pretrain-lvd1689m\model.safetensors)

Dataset:
[wdtaggerv4](X:\unique\datasets\animetimm-danbooru-wdtagger-v4-w640-ws-full\repo\README.md)
5914596 images in total
Split 	Image Count 	Total Size
train 	5321713 	318 GB
test 	295926 	17.7 GB
val 	296957 	17.8 GB

OFTv2 (A type of lora):
[OFTv2](C:\MyApps\OneTrainer\src)

Optimizer:
[SinkSGD_ADV](https://github.com/Koratahiu/Advanced_Optimizers)

# Training parameters
10,000 training pictures
1,000 validation pictures
1,000 test pictures


# Intended architecture
## Env
You can use python314 and install it to the $WS/.env/
The C:\Python314 already has the latest torch, env


## Caching:
    We need to cache the images as the trainer sees it
    First, rescale the chosen training images, test, and validation, to 512x512 resolution using bilinear scaling with resizing.
    Save each as their seperate folders as 0.5GB sized .pt chunked files: so a folder for training, a folder for validation, a folder for test, in `I:/.cache/wd-tagger-trainer/{train|test|valid}/`
    Name the .pt shard#####-{number of images in total for entire combined .pt}.pt
    Make the .pt creation pausable and resumable in case of error at the last shard
    The `/.cache/` folder indicates that you shouldn't expect this folder to 'live' and is only temporary, so it should be checked at run start, then if not exist, built at run start

## Training:
    I like to make my training pausable and resumable with .PUASE files on the root of the folder
    Training should output every 200 steps to $WS/runs/{sha256 8 character truncation of config}/{some name}.pt

    When resuming, take care to not show the same picture you would have for step 1, for the resumed steps

## Evaluation:
    Evaluations should output at the very end using the test pictures cache, to $WS/runs/{sha256 8 character truncation of config}/{someFile}

## Debug:
    you should always output the debug of the pictures as how the trainer saw it (as individual .png), for the first 100 pictures as $WS/runs/{sha256 8 character truncation of config}/debugpics/({pictureName}-{picture-TarOrigination}) <- or something like that, I will let you choose the descriptive picture name that is easy for user to trace back to the origination tar



## Folder names that may be helpful (but you don't have to use; taking inspiration from comfyUI)


api_server
app
blueprints
comfy
api
api_nodes
config
execution
extras
custom_nodes
input
middleware
models
output
script_examples
temp
tests
tests-uni
user
utils

# Goal

The goal is to beat [convnextv2_huge](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full)
[already downloaded here](C:\MyApps\ComfyUIRoot\convnextv2_huge.dbv4-full)

Beat it in our test of 1,000 test images.
Construct and evaluate convnextv2_huge against Macro(F1/MCC/P/R) ,Micro(F1/MCC/P/R) on our test of 1,000 images.

Then that will be our bar we have to beat. We probably won't beat it using only 10,000 training images, but it's worth a shot.