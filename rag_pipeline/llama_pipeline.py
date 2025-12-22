"""
llama_pipeline.py

Manages loading and inference of a LLaMA-based large language model using Hugging Face and LangChain.

Key Features:
- Automatically loads model on a specific device.
- Uses float16 precision for efficient performance on modern GPUs.
- Uses a system prompt template to generate recruiter-style evaluations.
- Truncates resume context to fit within the LLaMA model's context window.
"""

import re
import logging
import torch

from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from langchain_huggingface import HuggingFacePipeline


class LlamaPipeline:
    def __init__(
        self, 
        model_name: str, 
        huggingface_token: str, 
        max_new_tokens: int = 1024, 
        temperature: float = None, 
        top_p: float = 1.0
    ) -> None:
        """
        Initialize the LLaMA inference pipeline.
    
        Args:
            model_name (str): Identifier of the Hugging Face language model to use for response generation.
            huggingface_token (str): Authentication token for accessing Hugging Face models.
            max_new_tokens (int): Maximum number of tokens to generate in responses. 
            temperature (float): Sampling temperature controlling creativity. None disables sampling for deterministic output.
            top_p (float): Nucleus sampling probability threshold, controlling output diversity.
        """
        self.model_name = model_name
        self.huggingface_token = huggingface_token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.pipeline = None
        self.model_max_length = 4096

    def load(self) -> None:
        """
        Load the LLaMA model, tokenizer, and generation pipeline with quantization.
        """
        # Load the tokenizer associated with the model
        # This is responsible for converting text into tokens the model can understand
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.huggingface_token
        )

        # Define 4-bit quantization config for memory efficiency
        # - load_in_4bit=True to enable 4-bit weight quantization (saves GPU memory)
        # - bnb_4bit_compute_dtype=torch.float16 for fast and efficient compute
        # - bnb_4bit_use_double_quant=True to apply nested quantization for better compression
        # - bnb_4bit_quant_type="nf4" for high-accuracy quantization representation
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load the quantized model with automatic device mapping
        # - device_map to map the model to GPU
        # - quantization_config to apply 4-bit compression
        # - torch_dtype=torch.float16 for reduced memory usage and better performance
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map={"": "cuda:3"},
            quantization_config=bnb_config,
            token=self.huggingface_token,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Create a Hugging Face generation pipeline with sampling parameters
        # - max_new_tokens: limits the length of the response
        # - do_sample: disable randomness for deterministic answers
        # - temperature: control creativity/diversity of the response
        generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature is not None,
            temperature=self.temperature if self.temperature is not None else 1.0,
            top_p=self.top_p,
            pad_token_id=tokenizer.eos_token_id
        )

        # Wrap the generation pipeline using LangChain for integration with other components
        self.pipeline = HuggingFacePipeline(pipeline=generation_pipeline)

    def generate_response(self, question: str, resumes: str) -> str:
        """
        Generate an answer given a job description and relevant resumes.
    
        Args:
            question (str): The job description or sub-query.
            resumes (str): Combined text of top-matching resumes.
    
        Returns:
            str: Structured response from the LLaMA model.
        """
        # Ensure the LLaMA pipeline is loaded before use
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call `load()` first.")
    
        # 1. Get the tokenizer from the loaded pipeline
        tokenizer = self.pipeline.pipeline.tokenizer
    
        # 2. Define the system prompt with strict instructions for the LLM
        system_prompt = (
            "You are a meticulous technical recruiter. "
            "Your sole task is to determine the best candidate for a job based *only* on the evidence provided in the resumes.\n\n"
            "**Primary Directive: Grounding**\n"
            "Every single statement in your justification MUST be directly verifiable from the text in the candidate's resume. "
            "Do not infer, speculate, or embellish. If the resume does not explicitly state something, you cannot claim it.\n\n"
            "**Your Process:**\n"
            "1.  **Analyze Job Requirements:** Identify the key skills and experiences from the job description.\n"
            "2.  **Extract Evidence:** For each candidate, find verbatim phrases from their resume that match the job requirements.\n"
            "3.  **Select Best Fit:** Choose the single applicant with the strongest, most direct evidence.\n"
            "4.  **Justify:** Write your justification using only the evidence you extracted. Do not add any information not present in the resumes.\n\n"
            "**--- Bad Example (Speculation) ---**\n"
            "Justification:\n"
            "- The candidate is likely a strong leader because they managed a large project.\n"
            "*(This is an inference. The resume said 'managed a project,' not that they were a 'strong leader.')*\n\n"
            "**--- Good Example (Grounded) ---**\n"
            "Justification:\n"
            "- The candidate's resume states they 'Managed a project with a budget of $2M and a team of 5 engineers.'\n"
            "*(This is a direct, verifiable fact from the context.)*\n\n"
            "**Output Format (Strict):**\n"
            "- Selected Applicant ID: <Applicant ID>\n"
            "- Justification:\n"
            "  - <Direct, resume-based evidence 1>\n"
            "  - <Direct, resume-based evidence 2>"
        )

        # 3. Format the user-facing part of the prompt with the specific job and resume context
        user_prompt = (
            f"Job Description:\n{question}\n\n"
            f"Resumes:\n{resumes}\n\n"
        )

        # 4. Combine system and user prompts into the chat format expected by the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 5. Convert the chat messages into a single, formatted string prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
        # 6. Check token count and truncate the prompt if it exceeds the model's context window
        total_tokens = len(tokenizer.encode(prompt))
        if total_tokens > self.model_max_length:
            logging.warning(f"⚠️ Prompt exceeds max ({total_tokens} > {self.model_max_length}) → Truncating.")
            prompt_tokens = tokenizer.encode(prompt, truncation=True, max_length=self.model_max_length)
            prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    
        # 7. Send the final prompt to the model to get the generated response
        response = self.pipeline.invoke(prompt)

        # 8. Clean the response by removing the input prompt text, leaving only the generated answer
        return response.replace(prompt, "").strip()

    def generate_subqueries(self, job_description: str) -> List[str]:
        """
        Breaks a job description into 3-5 focused sub-queries to improve resume retrieval.
    
        Args:
            job_description (str): The job description.
    
        Returns:
            str: A list of 3-5 sub-queries.
        """
        # Ensure the LLaMA pipeline is loaded before use
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        # 1. Get the tokenizer from the loaded pipeline
        tokenizer = self.pipeline.pipeline.tokenizer

        # 2. Define the system prompt to instruct the model on how to decompose the job description
        system_prompt = (
            "You are an expert technical recruiter. Your task is to decompose a job description into 3-5 distinct, "
            "high-quality questions for a semantic search engine.\n\n"
            "Each question should:\n"
            "1. Be a full, natural language question.\n"
            "2. Focus on a specific aspect of the job, like 'technical skills,' 'team responsibilities,' or 'project experience.'\n"
            "3. Rephrase and combine concepts from the job description to capture the ideal candidate's profile.\n"
            "4. Be distinct from the other questions to ensure diverse search results.\n\n"
            "Exclude questions about location, salary, benefits, or company descriptions.\n"
            "Output each question on a new line, prefixed with 'SUBQUERY_X:'.\n\n"
            "--- Example ---\n"
            "Job Description:\n"
            "'The ideal candidate will have 5+ years of experience with Python and Django, building scalable web applications. "
            "They must have a deep understanding of REST APIs and PostgreSQL.'\n\n"
            "Output Questions:\n"
            "SUBQUERY_1: What is the candidate's experience in developing scalable web applications using Python and Django?\n"
            "SUBQUERY_2: How proficient is the candidate with designing and implementing REST APIs?\n"
            "SUBQUERY_3: Show me candidates with strong knowledge of PostgreSQL database management.\n"
            "--- End Example ---\n"
        )

        # 3. Provide the specific job description for the model to process
        user_instruction = (
            "Now apply the same logic to the following job description:\n\n"
            f"{job_description}\n\n"
        )

        # 4. Combine system and user prompts into the final chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_instruction}
        ]

        # 5. Convert the chat messages into a single, formatted string prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 6. Truncate the prompt if it exceeds the model's maximum length
        total_tokens = len(tokenizer.encode(prompt))
        if total_tokens > self.model_max_length:
            logging.warning(f"⚠️ Prompt too long ({total_tokens}), truncating.")
            prompt_tokens = tokenizer.encode(prompt, truncation=True, max_length=self.model_max_length)
            prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    
        # 7. Generate the raw text containing the subqueries from the model
        raw_output = self.pipeline.invoke(prompt)

        # 8. Isolate the generated part of the output by removing the input prompt
        generated_subqueries = raw_output.strip().replace(prompt, "")

        # 9. Parse the raw output to extract and clean each subquery

        # 9.1. Initialize a list to hold the final, cleaned subqueries
        subqueries = []
    
        # 9.2. Iterate through each line of the model's generated text
        for line in generated_subqueries.splitlines():
            # Use a regular expression to find lines that start with the "SUBQUERY_X:" pattern
            match = re.search(r"SUBQUERY_\d+:\s*(.*)", line)
            
            # If the line matches the expected format...
            if match:
                # ...extract the captured question (group 1)
                question = match.group(1).strip()
                
                # Ensure the extracted question is not an empty string
                if question:
                    # Add the clean, validated question to our list of subqueries
                    subqueries.append(question)
    
        # 10. Return the final list of processed subqueries
        return subqueries
