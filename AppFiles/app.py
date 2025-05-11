import chainlit as cl
from script import answer_user_query

@cl.on_message
async def main(message: cl.Message):
    # Run your analysis
    result = answer_user_query(message.content)
    
    # 1. First send the text analysis
    await cl.Message(content=result.get("text", "Sorry, no analysis available.")).send()
    
    # 2. Then send the Plotly chart if available
    chart_figure = result.get("chart_figure")
    if chart_figure:
        # Create a Plotly element - this will be interactive!
        plotly_chart = cl.Plotly(
            name="Stock Analysis", 
            figure=chart_figure,
            display="inline",
            size="large"
        )
        await cl.Message(content="", elements=[plotly_chart]).send()

        