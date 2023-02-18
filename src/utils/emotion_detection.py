import transformers


def get_emotion(text: str, classifier: transformers.pipelines) -> str:
    classifier_output = classifier(text)[0]
    classifier_output = {d['label']: d['score'] for d in classifier_output}
    emotion = max(classifier_output, key=classifier_output.get)
    return emotion
