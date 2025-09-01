import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# not sure about paddding_side (reference: https://github.com/huggingface/transformers/issues/34842) -> needs to be confirmed
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True, padding_side="left")
# not sure whether this process is necessary
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
# padding token id
pad_token_id = tokenizer.pad_token_id

# --- test code to gain mask indices ---
# example 1
# question = "What is the population of Tokyo?"
# knowledge = "The population of Tokyo is about 14 million."
# answer = "14 million"

# example 2 (the same format as my research datasets)
question = "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given one summary written for a news article(the following Source Text). Your task is to rate the summary on appropriate metrics. Source Text: Everton manager Roberto Martinez was forced to defend another penalty fiasco at the club after Ross Barkley missed from the spot in their 1-0 win against Burnley at Goodison Park. The untried Barkley inexplicably took the 10th minute kick \u2013 awarded for a foul by David Jones on Aaron Lennon \u2013 rather than Leighton Baines, who has scored 15 penalties from 16 attempts in the Premier League. Although there was no dispute between the team-mates this time, it brought back memories of Everton's match against West Brom in January when Kevin Mirallas grabbed the ball from Baines to take a penalty - and missed. Ross Barkley steps up to take a 10th minute penalty despite the presence of Leighton Baines on the pitch Barkley's effort is saved by\u00a0Burnley goalkeeper Tom Heaton at Goodison Park Martinez insisted Barkley was within his rights to request penalty-taking duties on Saturday. 'If Romelu Lukaku had been on the pitch, he would have taken it. Otherwise, I am happy to have three or four players who can take penalties and let it depend on how they feel at that moment,' argued the Everton manager. Baines (left)\u00a0has scored 15 penalties from 16 attempts in the Premier League 'Ross showed incredible responsibility to take it. I love seeing players take control of the big moments and Leighton was happy to given him that responsibility.' Barkley's penalty was well-struck but wasn't put in the corner and Burnley goalkeeper Tom Heaton dived to his right to save. Fortunately for the young England player, it didn't prove costly as Mirallas went on to score the only goal of the game after 29 minutes. Everton boss Roberto Martinez issues instructions to his players during a break in play against Burnley output: everton manager roberto martinez was forced to defend another penalty fiasco at the club after ross barkley missed from the spot in their 1 - 0 win against burnley at goodison park . the untried barkley inexplicably took the 10th minute kick \u2013 awarded for a foul by david jones on aaron lennon \u2013 rather than leighton baines , who has scored 15 penalties from 16 attempts in the premier league . martinez insisted barkley was within his rights to request penalty - taking duties on saturday .",
knowledge = "- **Coherence (4)**: The summary presents a clear narrative about the penalty incident involving Ross Barkley and Roberto Martinez's defense of it. However, it could improve by integrating more context about the match outcome and previous incidents for a fuller picture.\n\n- **Consistency (5)**: The summary accurately reflects the key facts from the source text, such as the missed penalty by Barkley and Martinez's comments, ensuring factual alignment without discrepancies.\n\n- **Fluency (5)**: The text is grammatically correct and flows well, making it easy to read and understand. There are no awkward constructions or errors that would hinder comprehension.\n\n- **Relevance (4)**: The summary includes crucial information about the penalty incident and Martinez's response but omits details about the match's final score and Barkley's performance, which could provide a more comprehensive understanding.\n\nAspects like creativity, empathy, and humor are not relevant to the task of summarizing factual information. Depth and inclusivity were not chosen"
answer = "{\"coherence\": 4, \"consistency\": 5, \"fluency\": 5, \"relevance\": 5}"
MAX_SEQ_LEN = 2048

# for qk_loss
input_ids_list_qk = []
attention_mask_list_qk = []
labels_start_indices_qk = [] # indices which starts knowledge parts
# for qa_loss
input_ids_list_qa = []
attention_mask_list_qa = []
labels_start_indices_qa = [] # indices which starts answer parts after question parts
# for qka_loss
input_ids_list_qka = []
attention_mask_list_qka = []
labels_start_indices_qka = [] # indices which starts answer parts after knowledge parts


# ==== 1. input construction and indices identification for QK Loss ====
# ---begin: just debug to confirm the function of apply_chat_template(this part will not be used for main process) ---
messages_qk_prompt = [{"role": "user", "content": question}]
# contain assistant start header by setting add_generation_prompt=True 
# reference: https://huggingface.co/docs/transformers/main/chat_templating 
prompt_string_qk = tokenizer.apply_chat_template(
    messages_qk_prompt,
    tokenize=False, # returns string here
    add_generation_prompt=True
)
print(f"Prompt_string_qk: {prompt_string_qk}")
# ---end: just debug to confirm the function of apply_chat_template(this part will not be used for main process) ---

# Combine the user's questions and the desired knowledge output into a single list.
messages_full_qk = [
    {"role": "user", "content": question},
    {"role": "assistant", "content": knowledge} # add knowledge here
]
# use apply_chat_template to full_text
full_text_qk = tokenizer.apply_chat_template(
    messages_full_qk,
    tokenize=False,
    add_generation_prompt=False  # template can automatically process the role of assistant
)
# tokenize full_text_qk using tokenizer(), and gain offsets_mapping
encoded_full_qk = tokenizer(
    full_text_qk,
    padding=False,
    truncation=True,
    max_length=MAX_SEQ_LEN,
    return_tensors='pt',
    add_special_tokens=True, # maybe can be False depending on model
    return_offsets_mapping=True
)

input_ids_list_qk.append(encoded_full_qk.input_ids.squeeze(0))
print(f"input_ids_list_qk: {input_ids_list_qk}")
attention_mask_list_qk.append(encoded_full_qk.attention_mask.squeeze(0))
print(f"attention_mask_list_qk: {attention_mask_list_qk}")
offsets_qk = encoded_full_qk.offset_mapping.squeeze(0)
print(f"offsets_qk: {offsets_qk}")

# idetify knowledge start indices from offsets_mapping
# search knowledge start indices in the template-applied text
knowledge_start_char_idx = full_text_qk.find(knowledge)
# error handling when we cannot find knowledge
if knowledge_start_char_idx == -1:
    raise ValueError("Knowledge not found in the formatted text.")

# search knowledge start token index using offsets_mapping
knowledge_start_token_idx = -1
for i, (start, end) in enumerate(offsets_qk):
    if start >= knowledge_start_char_idx:
        knowledge_start_token_idx = i
        break

print(f"knowledge_start_char_idx: {knowledge_start_char_idx}")
print(f"knowledge_start_token_idx: {knowledge_start_token_idx}")
print(f"Token at calculated start index: {tokenizer.convert_ids_to_tokens([encoded_full_qk.input_ids[0, knowledge_start_token_idx].item()])}")

# ==== 2. input construction and indices identification for QA Loss ====
# ---begin: just debug to confirm the function of apply_chat_template(this part will not be used for main process) ---
messages_qa_prompt = [{"role": "user", "content": question}]
prompt_string_qa = tokenizer.apply_chat_template(
    messages_qa_prompt,
    tokenize=False,
    add_generation_prompt=True 
)
print(f"Prompt_string_qa: {prompt_string_qa}")
# ---end: just debug to confirm the function of apply_chat_template(this part will not be used for main process) ---

# Combine the user's questions and the desired answer output into a single list.
messages_full_qa = [
    {"role": "user", "content": question},
    {"role": "assistant", "content": answer} 
]
# use apply_chat_template to full_text
full_text_qa = tokenizer.apply_chat_template(
    messages_full_qa,
    tokenize=False,
    add_generation_prompt=False
)
# tokenize full_text_qa using tokenizer(), and gain offsets_mapping
encoded_full_qa = tokenizer(
    full_text_qa,
    padding=False,
    truncation=True,
    max_length=MAX_SEQ_LEN,
    return_tensors='pt',
    add_special_tokens=True, 
    return_offsets_mapping=True
)

input_ids_list_qa.append(encoded_full_qa.input_ids.squeeze(0))
print(f"input_ids_list_qa: {input_ids_list_qa}")
attention_mask_list_qa.append(encoded_full_qa.attention_mask.squeeze(0))
print(f"attention_mask_list_qa: {attention_mask_list_qa}")
offsets_qa = encoded_full_qa.offset_mapping.squeeze(0)
print(f"offsets_qa: {offsets_qa}")

# idetify answer start indices from offsets_mapping
# search answer start indices in the template-applied text
answer_start_char_idx = full_text_qa.find(answer)
# error handling when we cannot find answer
if answer_start_char_idx == -1:
    raise ValueError("Answer not found in the formatted text.")

# search answer start token index using offsets_mapping
answer_start_token_idx = -1
for i, (start, end) in enumerate(offsets_qa):
    if start >= answer_start_char_idx:
        answer_start_token_idx = i
        break

print(f"answer_start_char_idx: {answer_start_char_idx}")
print(f"answer_start_token_idx: {answer_start_token_idx}")
print(f"Token at calculated start index: {tokenizer.convert_ids_to_tokens([encoded_full_qa.input_ids[0, answer_start_token_idx].item()])}")


# ==== 3. input construction and indices identification for QKA Loss ====
# ---begin: just debug to confirm the function of apply_chat_template(this part will not be used for main process) ---
messages_qka_prompt = [
    {"role": "user", "content": question},
    {"role": "assistant", "content": knowledge}, # assume knowledge was already generated by assistant
]
prompt_string_qka = tokenizer.apply_chat_template(
    messages_qka_prompt,
    tokenize=False,
    add_generation_prompt=True, 
)
print(f"Prompt_string_qka: {prompt_string_qka}")
# ---end: just debug to confirm the function of apply_chat_template(this part will not be used for main process) ---

# Combine the user's questions and the desired answer output into a single list.
messages_full_qka = [
    {"role": "user", "content": question},
    {"role": "assistant", "content": knowledge}, # assume knowledge was already generated by assistant
    {"role": "assistant", "content": answer} # add answer here
]

# use apply_chat_template to full_text
full_text_qka = tokenizer.apply_chat_template(
    messages_full_qka,
    tokenize=False,
    add_generation_prompt=False
)
# tokenize full_text_qa using tokenizer(), and gain offsets_mapping
encoded_full_qka = tokenizer(
    full_text_qka,
    padding=False,
    truncation=True,
    max_length=MAX_SEQ_LEN,
    return_tensors='pt',
    add_special_tokens=True,
    return_offsets_mapping=True
)

input_ids_list_qka.append(encoded_full_qka.input_ids.squeeze(0))
print(f"input_ids_list_qka: {input_ids_list_qka}")
attention_mask_list_qka.append(encoded_full_qka.attention_mask.squeeze(0))
print(f"attention_mask_list_qka: {attention_mask_list_qka}")
offsets_qka = encoded_full_qka.offset_mapping.squeeze(0)
print(f"offsets_qka: {offsets_qka}")

# idetify answer start indices from offsets_mapping
# search answer start indices in the template-applied text
answer_start_char_idx = full_text_qka.find(answer)
# error handling when we cannot find answer
if answer_start_char_idx == -1:
    raise ValueError("Answer not found in the formatted text.")

# search answer start token index using offsets_mapping
answer_start_token_idx = -1
for i, (start, end) in enumerate(offsets_qka):
    if start >= answer_start_char_idx:
        answer_start_token_idx = i
        break

print(f"answer_start_char_idx: {answer_start_char_idx}")
print(f"answer_start_token_idx: {answer_start_token_idx}")
print(f"Token at calculated start index: {tokenizer.convert_ids_to_tokens([encoded_full_qka.input_ids[0, answer_start_token_idx].item()])}")


# ==== 4. padding for the whole batch ====
# pading fitted for the max sequence of batch at the end of collate_fn
# it is conveient to use tokenizer.pad() 
# for qk_loss
padded_batch_qk = tokenizer.pad(
    {'input_ids': input_ids_list_qk, 'attention_mask': attention_mask_list_qk},
    padding='longest', # pading fitted for the max sequence of batch
    return_tensors='pt',
    pad_to_multiple_of=8 # Consider padding to multiples of 8 for training efficiency
)

# for qa_loss
padded_batch_qa = tokenizer.pad(
    {'input_ids': input_ids_list_qa, 'attention_mask': attention_mask_list_qa},
    padding='longest', # pading fitted for the max sequence of batch
    return_tensors='pt',
    pad_to_multiple_of=8 # Consider padding to multiples of 8 for training efficiency
)

# for qka_loss
padded_batch_qka = tokenizer.pad(
    {'input_ids': input_ids_list_qka, 'attention_mask': attention_mask_list_qka},
    padding='longest', # pading fitted for the max sequence of batch
    return_tensors='pt',
    pad_to_multiple_of=8 # Consider padding to multiples of 8 for training efficiency
)

# memo: I use start_token_idx, input_ids and attention_mask in my research Trainer