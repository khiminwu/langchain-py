# 🔹 Custom Tool 1: SWOT Analysis
swot_prompt = PromptTemplate(
    input_variables=["brand", "audience"],
    template="""
Perform a SWOT analysis for a coffee shop brand.

Brand: {brand}
Target Audience: {audience}

Return in 4 sections: Strengths, Weaknesses, Opportunities, Threats.
"""
)

swot_chain = LLMChain(llm=llm, prompt=swot_prompt)