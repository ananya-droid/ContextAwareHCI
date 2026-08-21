from communication_engine import CommunicationEngine


engine = CommunicationEngine()

engine.add_token("HELLO")
engine.add_token("I")
engine.add_token("NEED")
engine.add_token("HELP")

print("Current tokens:")
print(engine.tokens)

print("\nCurrent text:")
print(engine.get_text())

print("\nFinal sentence:")
print(engine.finish_sentence())