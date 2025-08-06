## Step-by-Step Guide to Create Your Dataset for TTS Model Training

This guide will help you prepare and format your dataset for training a TTS model, ensuring that all files and folders are set up correctly for easy integration with the training pipeline.

## Table of Contents

- [Step-by-Step Guide to Create Your Dataset for TTS Model Training](#step-by-step-guide-to-create-your-dataset-for-tts-model-training)
  - [1. Directory Structure](#1-directory-structure)
  - [2. Preparing `metadata_train.csv` and `metadata_eval.csv`](#2-preparing-metadatatraincsv-and-metadataevalcsv)
  - [3. Common Formatting Mistakes to Avoid](#3-common-formatting-mistakes-to-avoid)
  - [4. Example Walkthrough for Creating CSV Files](#4-example-walkthrough-for-creating-csv-files)
  - [5. Using the dataset and running training](#5-using-the-dataset-and-running-training)

## **1. Directory Structure**

To start, create a structured directory for your TTS dataset. You will need a folder that contains all necessary files and subdirectories to properly organize your data.

**Folder Structure:**
- Create a project folder inside your main path (for example, `/alltalk_tts/finetune/`).
- Name this folder with a project name, such as `myproject`.

Inside the project folder, the structure should look like this:

```
📁 finetune/
└── 📁 myproject/
    ├── 🗎 metadata_train.csv
    ├── 🗎 metadata_eval.csv
    ├── 🗎 lang.txt
    └── 📁 wavs/
        ├── 🗣️ audio1.wav
        ├── 🗣️ audio2.wav
        └── 🗣️....
```

**Explanation:**
- **`metadata_train.csv`**: This is a CSV file that contains metadata information for the training set.
- **`metadata_eval.csv`**: This is another CSV file that contains metadata for evaluation purposes.
- **`lang.txt`**: A file with the two-letter language code (e.g., `en` for English).
- **`wavs/`**: A subfolder containing all the audio files (`.wav` format) used for both training and evaluation.


### Example: `metadata_train.csv`

The `metadata_train.csv` contains the main training dataset and should have three columns, separated by pipes (`|`). Here is a sample of what the training file might look like:

```
audio_file|text|speaker_name
wavs/audio1.wav|The quick brown fox jumps over the lazy dog.|projectname.
wavs/audio2.wav|She said, "Good morning!" and waved.|projectname.
wavs/audio3.wav|It's 5 o'clock somewhere.|projectname.
wavs/audio4.wav|The temperature today is 20°C.|projectname.
wavs/audio5.wav|You'll need about $30 to buy that.|projectname.
```

### Example: `metadata_eval.csv`

The `metadata_eval.csv` contains data that will be used for evaluating the model’s performance. It should have the same format as the training CSV file, with three columns separated by pipes (`|`). An Eval file will typically list 10-15% of your total amount audio files and you should **not** cross pollinate audio files between the `eval` and the `train` CSV files (do not share the same audio clips between them). Below is an example:

```
audio_file|text|speaker_name
wavs/audio6.wav|How much is 5 plus 10?|projectname
wavs/audio7.wav|It's 3 PM now.|projectname
wavs/audio8.wav|They'll arrive at 7 o'clock.|projectname.
wavs/audio9.wav|My favorite year was 1995.|projectname
wavs/audio10.wav|The cat chased the mouse.|projectname
```


**Explanation of Rows**:
1. **First Column**: The filename of the audio file located in the `wavs/` folder, e.g., `wavs/audio1.wav`.
2. **Second Column**: The **raw transcription** that matches exactly what is spoken in the audio.
3. **Third Column**: The **Name of the person/project you wish to train** e.g. projectname
4. **The very first row**: Should always have the text `audio_file|text|speaker_name` inside it.

## **2. Preparing `metadata_train.csv` and `metadata_eval.csv`**

The two CSV files, `metadata_train.csv` and `metadata_eval.csv`, contain the main dataset information, with details about each audio file and its transcription.

**Structure of CSV Files:**
- The CSV files should have **three columns** for each audio file. These columns are:
  1. **Audio Filename**: The name of the audio file.
  2. **Raw Transcription**: The exact text spoken in the audio.
  3. **Third Column**: The **Name of the person/project you wish to train** e.g. projectname

First row in the CSV file should have the following text:
```
audio_file|text|speaker_name
```

Ehen in each row in the CSV file should follow the format:
```
wavs/<filename>|<raw transcription>|<project name>
```
Here, the **pipe (`|`)** character is used to separate each of these columns.

**Important Notes for Proper Formatting:**
- **Filename**: Only the name of the audio file, not including the full path. For example, `wavs/audio1.wav`.
- **Pipe Delimiter (`|`)**: The pipe character (`|`) is used to separate the **three columns**. It is crucial that this character is not used inside the transcription text itself. It serves only as a **column separator**.
- **Punctuation in Transcriptions**: Use natural punctuation marks like commas, periods, and question marks where needed within the **transcriptions** to accurately represent the spoken content of the audio. However, these punctuation marks should not interfere with the columns.
- **No Extra Columns**: Each row should have exactly **three columns** separated by **two pipe characters (`|`)**. No more, no less.

**Example of CSV Formatting:**
- Correctly formatted rows:
  1. `wavs/audio1.wav|The quick brown fox jumps over the lazy dog.|projectname`
  2. `wavs/audio2.wav|She said, "Hello there!"|projectname`
  3. `wavs/audio3.wav|It will cost $50.|projectname`

**Detailed Breakdown:**
1. **Filename**: 
   - Should exactly match the `.wav` file located in the `wavs/` folder.
   - Example: `wavs/audio1.wav`.
   
2. **Raw Transcription**:
   - Represents exactly what is said in the audio file, including pauses, emphasis, and punctuation.
   - Use **natural punctuation** (commas, periods, dollar signs, etc.) to accurately transcribe the spoken words.

3. **Project Name**:

**Do’s and Don’ts for Preparing the CSV:**
- **DO**:
  - Use the **pipe (`|`)** as the delimiter **between columns**.
  - Use natural punctuation (commas, periods) **inside transcriptions** as needed.
  - Ensure there are **exactly three columns** for each row.
  - Keep the filename accurate, as it is a direct reference to the audio file in the `wavs/` folder.

- **DON'T**:
  - Do not use **pipes (`|`)** inside the transcription text. This will confuse the training process.
  - Do not add any extra columns or data outside the three required columns.
  - Avoid leaving **any column empty**. If there is no specific normalized transcription, just copy the raw transcription.

## **3. Common Formatting Mistakes to Avoid**

1. **Incorrect Number of Pipes**: There should be **exactly two pipes (`|`)** in each row—one between each of the three columns. If there are more or fewer pipes, the CSV file will not be parsed correctly.
  
2. **Missing or Incorrect Filenames**: Ensure the filenames in the first column match **exactly** to the names of the `.wav` files in the `wavs/` directory. They are case-sensitive, so `wavs/audio1.wav` and `wavs/Audio1.wav` would be considered different.

3. **Excel Issues**: If you are using Excel to edit the CSV, it might not support using `|` as a delimiter by default. You may need to specify this when importing or exporting the data. Alternatively, you can use a plain text editor like **Notepad** or **VS Code** to manually check the format.

## **4. Example Walkthrough for Creating CSV Files**

- **Step 1**: Gather all `.wav` files and place them into the `wavs/` folder in your project directory.
- **Step 2**: Use a **spreadsheet editor** (like Google Sheets or Excel) or a **text editor** (like Notepad++) to create your CSV files.
- **Step 3**: Each row should contain:
  - **Column 1**: The **filename** of the audio file (e.g., `wavs/audio1.wav`).
  - **Column 2**: The **exact spoken text** from the audio (e.g., `Hello, how are you?`).
  - **Column 3**: A **projectname** (e.g., `elonmusk`, `annefrank`, `myproject`).
- **Step 4**: Save the file as a `.csv`, ensuring that each row follows the `<filename>|<raw transcription>|<project name>` pattern.
- **Step 5**: If editing manually, make sure there are no extra spaces around the pipes and each row ends correctly without trailing columns or delimiters.

**Example of How a Full Row Should Look:**
- **Correct**: `wavs/audio1.wav|This is an example sentence.|projectname`
- **Incorrect**: `audio1.wav|This is an example sentence|` (missing one column), or `audio1.wav, This is an example sentence, This is an example sentence` (incorrect use of commas instead of pipes).

## **5. Using the dataset and running training**

Ensure your folder structure is laid out in the correct standard e.g.

```
📁 finetune/
└── 📁 myproject/
    ├── 🗎 metadata_train.csv
    ├── 🗎 metadata_eval.csv
    ├── 🗎 lang.txt
    └── 📁 wavs/
        ├── 🗣️ audio1.wav
        ├── 🗣️ audio2.wav
        └── 🗣️....
```

Start Finetuning and in the interface, go to Step 2 and populate:

- **Project Name:** the name of the project, aka the 3rd column you used in your CSV files
- **Train CSV file path:** pointed to the full path to `metadata_train.csv`
- **Eval CSV file path:** pointed to the full path to `metadata_eval.csv`

Set all your other settings as you wish to use and click **"Run the Training"**

![image](https://github.com/user-attachments/assets/fec6eac3-ff21-4709-8942-bfe020380a2e)
