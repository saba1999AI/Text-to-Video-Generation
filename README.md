


### Outputs

<table>
  <tr>
    <td> <img src="https://github.com/user-attachments/assets/250a30f0-10db-4750-a81b-e3ea86b00687"  alt="1" width = 960px height = 360px ></td>

   
 <video src="https://github.com/user-attachments/assets/cc5a869c-4852-421f-b1a0-71a877635087.mp4" width="300" />

<td>Prompt : "A glamorous portrait of an elegant woman in a sparkling navy blue gown with intricate detailing. She exudes sophistication, wearing bold red lipstick and dazzling diamond-shaped earrings, set against a blurred background of a prestigious event."</td>

   </tr> 
   <tr>
      <td><img src="https://github.com/user-attachments/assets/2d70afbd-f80c-4e53-8331-4e49535424c0" alt="3" width = 960px height = 360px></td>

<video src="https://github.com/user-attachments/assets/83f3ff3f-2721-4fa6-8d86-2be470fdd6f4.mp4" width="300" />
     
  <td>Prompt : "A stunning close-up portrait of an elegant woman dressed in a strapless white gown, adorned with layered silver and black necklaces. Her makeup highlights bold red lips and a sharp cat-eye, complemented by soft waves of golden hair cascading over her shoulder. The background features subtle hints of a glamorous event with blurred golden accents."

</td>
  </tr>
</table>


<h5 align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/ConsisID-jupyter/blob/main/ConsisID_jupyter.ipynb)


</h5>




```bash
python app.py
```

### CLI Inference

```bash
python infer.py --model_path BestWishYsh/ConsisID-preview
```

warning: It is worth noting that even if we use the same seed and prompt but we change a machine, the results will be different.

### GPU Memory Optimization

```bash
# turn on if you don't have multiple GPUs or enough GPU memory(such as H100)
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

warning: it will cost more time in inference and may also reduce the quality.

### Prompt Refiner

ConsisID has high requirements for prompt quality. You can use [GPT-4o](https://chatgpt.com/) to refine the input text prompt, an example is as follows (original prompt: "a man is playing guitar.")
```bash
a man is playing guitar.

Change the sentence above to something like this (add some facial changes, even if they are minor. Don't make the sentence too long): 

The video features a man standing next to an airplane, engaged in a conversation on his cell phone. he is wearing sunglasses and a black top, and he appears to be talking seriously. The airplane has a green stripe running along its side, and there is a large engine visible behind his. The man seems to be standing near the entrance of the airplane, possibly preparing to board or just having disembarked. The setting suggests that he might be at an airport or a private airfield. The overall atmosphere of the video is professional and focused, with the man's attire and the presence of the airplane indicating a business or travel context.
```

Some sample prompts are available [here](https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/prompt.xlsx).

## ⚙️ Requirements and Installation

We recommend the requirements as follows.

### Environment

```bash
git clone --depth=1 https://github.com/saba1999AI/Text-to-Video-Generation
cd ConsisID
conda create -n consisid python=3.11.0
conda activate consisid
pip install -r requirements.txt
```

### Download weights

The weights are available at [🤗HuggingFace](https://huggingface.co/BestWishYsh/ConsisID-preview) and [🟣WiseModel](https://wisemodel.cn/models/SHYuanBest/ConsisID-Preview/file), and will be automatically downloaded when runing `app.py` and `infer.py`, or you can download it with the following commands.

```bash
# way 1
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
cd util
python download_weights.py

# way 2
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type model \
BestWishYsh/ConsisID-preview \
--local-dir ckpts

# way 3
git lfs install
git clone https://www.wisemodel.cn/SHYuanBest/ConsisID-Preview.git
```

Once ready, the weights will be organized in this format:

```
📦 ckpts/
├── 📂 data_process/
├── 📂 face_encoder/
├── 📂 scheduler/
├── 📂 text_encoder/
├── 📂 tokenizer/
├── 📂 transformer/
├── 📂 vae/
├── 📄 configuration.json
├── 📄 model_index.json
```

## 🗝️ Training

### Data preprocessing

Please refer to [this guide](https://github.com/PKU-YuanGroup/ConsisID/tree/main/data_preprocess) for how to obtain the [training data](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data) required by ConsisID. If you want to train a text to image and video generation model. You need to arrange all the dataset in this [format](https://github.com/PKU-YuanGroup/ConsisID/tree/main/asserts/demo_train_data/dataname):

```
📦 datasets/
├── 📂 captions/
│   ├── 📄 dataname_1.json
│   ├── 📄 dataname_2.json
├── 📂 dataname_1/
│   ├── 📂 refine_bbox_jsons/
│   ├── 📂 track_masks_data/
│   ├── 📂 videos/
├── 📂 dataname_2/
│   ├── 📂 refine_bbox_jsons/
│   ├── 📂 track_masks_data/
│   ├── 📂 videos/
├── ...
├── 📄 total_train_data.txt
```

### Video DiT training

First, setting hyperparameters:

- environment (e.g., cuda): [deepspeed_configs](https://github.com/PKU-YuanGroup/ConsisID/tree/main/util/deepspeed_configs)
- training arguments (e.g., batchsize): [train_single_rank.sh](https://github.com/PKU-YuanGroup/ConsisID/blob/main/train_single_rank.sh) or [train_multi_rank.sh](https://github.com/PKU-YuanGroup/ConsisID/blob/main/train_multi_rank.sh)

Then, we run the following bash to start training:

```bash
# For single rank
bash train_single_rank.sh
# For multi rank
bash train_multi_rank.sh
```


## 🐳 Dataset

We release the subset of the data used to train ConsisID. The dataset is available at [HuggingFace](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data), or you can download it with the following command. Some samples can be found on our [Project Page](https://pku-yuangroup.github.io/ConsisID/).

```bash
huggingface-cli download --repo-type dataset \
BestWishYsh/ConsisID-preview-Data \
--local-dir BestWishYsh/ConsisID-preview-Data
```


