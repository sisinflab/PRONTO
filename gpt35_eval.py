import numpy as np
import time
from openai import OpenAI
client = OpenAI(api_key="INSERT_KEY")
from tqdm import tqdm
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
SYSTEM_PROMPT = "You are a binary classifier. Your role is to predict whether the following statements are true or false. Answer with either true or false."
token_count = len(encoding.encode(SYSTEM_PROMPT))

def get_completion(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=5,
    )
    return response.choices[0].message.content

curr_acc = 0
questions = []
responses = []
total_tokens = 0

data_neg = np.loadtxt("./test_pairs.txt", delimiter="\t", dtype=str)
data_neg_gt = np.loadtxt("./test_pairs_gt.txt", delimiter="\t", dtype=str)
with tqdm(total=len(data_neg)) as pbar:
    for i, pair in enumerate(data_neg):
        prompt = f"{pair[0]} is a kind of {pair[1]}"
        response = get_completion(prompt)
        print(prompt)
        total_tokens = total_tokens + token_count + len(encoding.encode(prompt))
        print(response)
        curr_acc = curr_acc + ("true" in response.lower()) == ("1.0" in data_neg_gt[i])

        questions.append(prompt)
        responses.append(response)
        pbar.update(1)
        pbar.set_postfix(
            curr_acc=str(curr_acc / (i+1)),
            total_tokens=total_tokens
        )
        time.sleep(2)

with open("result.txt", "w") as output:
    output.write(str(list(zip(questions, responses))))

print("Acc: ", curr_acc / (i+1))

