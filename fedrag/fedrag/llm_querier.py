"""fedrag: A Flower Federated RAG app."""

import os
import re
from openai import OpenAI


class LLMQuerier:

    def __init__(self, model_name="gpt-4.1", use_gpu=False):
        # Initialize OpenAI client
        self.client = OpenAI()
        self.model_name = model_name
        # use_gpu parameter is ignored for OpenAI API but kept for compatibility

    def answer(self, question, documents, options, dataset_name, max_new_tokens=500):
        # Format options as A) ... B) ... etc.
        #print(f"Question: {question}")
        #print(f"Documents: {documents}")
        formatted_options = "\n".join([f"{k}) {v}" for k, v in options.items()])

        prompt = self.__format_prompt(
            question, documents, formatted_options, dataset_name
        )

        try:
            # Use OpenAI API to generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful medical expert assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            generated_text = response.choices[0].message.content
            print(f"Generated text: {generated_text}")
            return prompt, generated_text
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return prompt, None

    @classmethod
    def __format_prompt(cls, question, documents, options, dataset_name):
        if dataset_name == "pubmedqa":
            instruction = "As an expert doctor in clinical science and medical knowledge, can you tell me if the following statement is correct? Answer yes, no, or maybe."
        elif dataset_name == "bioasq":
            instruction = "You are an advanced biomedical AI assistant trained to understand and process medical and scientific texts. Given a biomedical question, your goal is to provide a concise and accurate answer based on relevant scientific literature."
        elif dataset_name == "medical":
            instruction = "You are a helpful medical expert. Based on the provided medical documents, answer the question clearly and concisely. Provide specific medical information when available."
        else:
            instruction = "You are a helpful medical expert, and your task is to answer a medical question using the relevant documents."

        ctx_documents = "\n".join(
            [f"Document {i + 1}: {doc}" for i, doc in enumerate(documents)]
        )
        
        # For interactive mode (medical dataset), use a more natural format
        if dataset_name == "medical":
            prompt = f"""{instruction}

Here are the relevant medical documents:
{ctx_documents}

Question: {question}

Answer: """
        else:
            prompt = f"""{instruction}

            Here are the relevant documents:
            {ctx_documents}

            Question: 
            {question}

            Options:
            {options}

            Answer only with the correct option: """
        return prompt

    @classmethod
    def __extract_answer(cls, generated_text, original_prompt):
        # For OpenAI API, we get the response directly without the original prompt
        response = generated_text.strip()

        # For multiple choice questions, find first occurrence of A-D (case-insensitive)
        option = re.search(r"\b([A-Da-d])\b", response)
        if option:
            return option.group(1).upper()
        
        # For interactive mode, return the full response (up to 500 characters for better answers)
        if response:
            return response[:500] + "..." if len(response) > 500 else response
        
        return None
