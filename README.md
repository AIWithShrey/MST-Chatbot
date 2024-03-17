# MST-Chatbot
LLM-powered chatbot for MS&amp;T.

ChatACM is a chatbot developed by Shreyas Mocherla, a Junior in Computer Science studying at MS&T.

## Features:

- ChatACM can give you general information as well as questions regarding MS&T specifically.
- It can browse the web on command and retrieve up-to-date results.
- The most recent feature includes the generation of images, powered by the [SDXL](https://stablediffusionxl.com) model trained by [Stability.ai](https://stability.ai). The chatbot smartly decides when to call an API to the hosted SDXL model on [Replicate](https://replicate.com/stability-ai/sdxl) and produces a result based on the user's request.

## Known-bugs:

- Ungraceful error handling. The chatbot throws a traceback error when it runs into an issue.
- Repeated prompts for desired outcomes. The chatbot will need nudging more than a single prompt to achieve desired results as a response.
- Misinformation even after web search. Sometimes the chatbot will search the web for results and come back with incorrect results. In this case, rephrase your question and observe the quality of the new response.

## Future features:

- Nothing new planned as of now.
- Submit future ideas here: [ChatACM future ideas](https://forms.gle/T6fGyAVCB5nEUWNX6)