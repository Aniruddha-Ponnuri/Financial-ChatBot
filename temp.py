# def split_content(content, max_length=6000):
#     return [
#         content[i:i + max_length] for i in range(0, len(content), max_length)
#     ]

# def parse_with_groq(dom_chunks, parse_description):
#     parsed_results = []

#     for i, chunk in enumerate(dom_chunks, start=1):
#         try:
#             input_prompt = f"Parse the following content:\n\"\"\"{chunk}\"\"\"\n\n{parse_description}"
#             response = groq_client.chat.completions.create(
#                 messages=[{"role": "user", "content": input_prompt}],
#                 model="llama-3.1-8b-instant"
#             )
#             parsed_content = response.choices[0].message.content.strip()
#             parsed_results.append(parsed_content)
            
#             logging.info(f"Parsed batch: {i} of {len(dom_chunks)}")
        
#         except Exception as e:
#             logging.error(f"Error parsing chunk {i}: {str(e)}")
#             parsed_results.append(f"Error parsing chunk {i}: {str(e)}")

#     return "\n".join(parsed_results)