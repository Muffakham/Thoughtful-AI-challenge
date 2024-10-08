THRESHOLD = 0.2
QUESTIONS = [
    "What does the eligibility verification agent (eva) do?",
    "What does the claims processing agent (cam) do?",
    "How does the payment posting agent (phil) work?",
    "Tell me about Thoughtful AI's agents.",
    "What are the benefits of using Thoughtful AI's agents?"
]

QA_DATASET = {
    QUESTIONS[0]: (
        "EVA automates the process of verifying a patient’s eligibility and benefits "
        "information in real-time, eliminating manual data entry errors and reducing claim rejections."
    ),
    QUESTIONS[1]: (
        "CAM streamlines the submission and management of claims, improving accuracy, "
        "reducing manual intervention, and accelerating reimbursements."
    ),
    QUESTIONS[2]: (
        "PHIL automates the posting of payments to patient accounts, ensuring fast, "
        "accurate reconciliation of payments and reducing administrative burden."
    ),
    QUESTIONS[3]: (
        "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. "
        "These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
    ),
    QUESTIONS[4]: (
        "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, "
        "and reduce errors in critical processes like claims management and payment posting."
    )
}

# Fallback response
FALLABCK_RESPONSE = "I'm sorry, but I don't have an answer for that. Please check our website for more information."

# Default response
DEFAULT_RESPONSE = "Welcome to Thoughtful AI! Feel free to ask me any questions about our services and automation agents."
