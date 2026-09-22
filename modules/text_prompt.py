import torch
import clip

def text_prompt(data, device="cpu"):
    # text_aug = ['{}']
    text_aug = ['a video of a person {}.']

    # Kinetics
    # text_aug = [
    #     'a photo of a person {}.',
    #     'a photo of {}.',
    #     'a photo of a person using {}.',
    #     'a photo of a person doing {}.',
    #     'a photo of a person during {}.',
    #     'a photo of a person performing {}.',
    #     'a photo of a person practicing {}.',
    #     'a video of {}.',
    #     'a video of a person {}.',
    #     'a video of a person using {}.',
    #     'a video of a person doing {}.',
    #     'a video of a person during {}.',
    #     'a video of a person performing {}.',
    #     'a video of a person practicing {}.',
    #     'a example of {}.',
    #     'a example of a person {}.',
    #     'a example of a person using {}.',
    #     'a example of a person doing {}.',
    #     'a example of a person during {}.',
    #     'a example of a person performing {}.',
    #     'a example of a person practicing {}.',
    #     'a demonstration of {}.',
    #     'a demonstration of a person {}.',
    #     'a demonstration of a person using {}.',
    #     'a demonstration of a person doing {}.',
    #     'a demonstration of a person during {}.',
    #     'a demonstration of a person performing {}.',
    #     'a demonstration of a person practicing {}.',
    # ]

    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])

    classes = text_dict[0].to(device)

    return classes, num_text_aug, text_dict

def get_tokenized_captions(data, device="cpu", tokenizer=None):
    text_aug = ['a video of {}.']

    tokenized_captions = {key: [] for key in data.caption_list.keys()}

    for idx, txt in enumerate(text_aug):
        for cls, captions in data.caption_list.items():
            for caption in captions:
                try:
                    if tokenizer:
                        tokenized_captions[cls].append(tokenizer(txt.format(caption), return_tensors="pt").to(device))
                    else:
                        tokenized_captions[cls].append(clip.tokenize(txt.format(caption)).to(device))
                except:
                    continue

    return tokenized_captions