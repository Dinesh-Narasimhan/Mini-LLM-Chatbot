# Mini-LLM-Chatbot

PROJECT: Multilingual Chat Decoder using Transformer Architecture
________________________________________
1. PROJECT TITLE
Building a Multilingual Chatbot Decoder from Scratch using Transformer Architecture
________________________________________
2. OBJECTIVE
To build a Transformer-based decoder model capable of generating multilingual chatbot responses trained only on custom data across four languages (English, Hindi, Telugu, Tamil).
________________________________________
3. DATASET PIPELINE
3.1 Data Collection & Extraction
We used four language datasets from custom chat data:

•	Hindi-English

•	Telugu-English

•	Tamil-English

•	Plain English

Data Preprocessing Steps:

•	Raw chat data from various documents were extracted.

•	Irregular formatting and unwanted symbols were removed.

•	Each chat line was cleaned, and user-bot dialogues were aligned properly.

3.2 Combining Text

•	The cleaned files from the four languages were concatenated into one .txt file: cleaned_combined_chat_data.txt.

•	This file serves as the foundation for tokenizer training.

3.3 Byte Pair Encoding (BPE)

•	BPE was used to create a tokenizer that learns frequent subwords rather than individual characters.

Technical Example:

Input sentence:

What are you doing?

BPE Tokenizer Output:

['W', 'hat', 'are', 'you', 'doing', '?']

Each subword is mapped to a token ID. For example:

['W', 'hat', 'are', 'you', 'doing', '?']     →       [134, 79, 23, 97, 118, 3] 

 

 Tokenizer Illustration:
 
Text  →  [Subwords]  →  [Token IDs]  →  [Vectors]

![image](https://github.com/user-attachments/assets/31b88d01-ec42-4821-9293-d1346731e44d)

________________________________________
4. TOKENIZATION TO ENCODING
Each sentence is tokenized and then encoded to token IDs. These token IDs are then converted to embedding vectors for model input.
Technical Example:
 Convert to token IDs
ids = tokenizer.encode("What are you doing?").ids

 Convert to tensors
tensor = torch.tensor(ids)  # Shape: [seq_length]
________________________________________
5. MODEL ARCHITECTURE
The model is a Transformer Decoder built from scratch using PyTorch.

5.1 Architecture Overview:

•	Token Embedding

•	Positional Embedding

•	6 Transformer Decoder Blocks:

  o	Multi-Head Self-Attention (8 heads)

  o	LayerNorm

  o	FeedForward Network

  o	Dropout

•	Final LayerNorm

•	Linear Output Head

5.2 Architecture Diagram:

![image](https://github.com/user-attachments/assets/33f45881-9c75-4c27-80bb-f3cfe15bd0af)

                              
Model Size:

•	Parameters: ~29.29M

•	Vocab Size: 10,000

•	Layers: 6

•	Heads: 8

•	Embedding Dim: 512

 This qualifies as a Mini LLM suitable for targeted multilingual chatbot deployment!
________________________________________
6. TRAINING PIPELINE

6.1 Dataset Preparation

•	After BPE tokenization, the sequences were saved as:
np.savez("chat_data_sequences.npz", inputs=inputs, targets=targets)

6.2 Training Setup

model = TransformerDecoderModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

6.3 Loss Function

•	CrossEntropyLoss (ignoring padding token 0)

•	Input and Target tensors are flattened:

logits = logits.reshape(-1, vocab_size)

targets = targets.reshape(-1)

6.4 Output

Model saved as:

torch.save(model.state_dict(), "trained_decoder_model.pth")
________________________________________
7. GENERATION MODULE (generate.py)

The model is evaluated using a generate loop.

input_ids = tokenizer.encode(prompt).ids

for _ in range(max_length):

    logits = model(input_tensor)    
    
    next_token = sample_from(logits)
    
    input_tensor = torch.cat([input_tensor, next_token], dim=1)
________________________________________
8. STREAMLIT DEPLOYMENT

8.1 Interface Setup

st.title("Multilingual Chatbot")

user_input = st.text_input("You:")

if st.button("Generate"):

    response = generate_text(user_input)
    
    st.write(f"Bot: {response}")

8.2 Hosting

•	Run using streamlit run app.py

•	Can be deployed on platforms like Streamlit Cloud or Vercel

Output:

![image](https://github.com/user-attachments/assets/981c6747-7008-43d0-b5ef-4b94a1a9e893)


________________________________________
9. CONCLUSION

This multilingual chatbot decoder is custom-built from scratch using a transformer decoder architecture, trained solely on custom chat data in 4 languages. It is lightweight, flexible, and suited for edge use-cases like low-resource NLP applications.

 It is a Mini LLM, trained only on task-specific multilingual chatbot data, and small enough to run without cloud GPUs.
________________________________________
Prepared by: Dinesh Narasimhan

Technologies Used: PyTorch, Tokenizers, NumPy, Streamlit

