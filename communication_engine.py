# communication_engine.py


class CommunicationEngine:

    def __init__(self):
        self.tokens = []

    def add_token(self, token):

        if token != "UNKNOWN" and token != "NO HAND":
            self.tokens.append(token)

    def get_text(self):

        return " ".join(self.tokens)

    def clear(self):

        self.tokens = []

    def finish_sentence(self):

        text = self.get_text()

        if not text:
            return ""

        # Simple rule-based sentence cleanup
        sentence = text.capitalize()

        if not sentence.endswith((".", "!", "?")):
            sentence += "."

        return sentence