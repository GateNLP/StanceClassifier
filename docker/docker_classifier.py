# Logic to configure and load the correct classifier for the Docker container
# based on environment variable settings

import os

# STANCE_CLASSIFIER_MODE should be set to "oblivious" for target-oblivious mode,
# "aware" for target-aware mode with a single model, or "ensemble" for target-aware
# mode using an ensemble of two models
classifier_mode = os.environ.get("STANCE_CLASSIFIER_MODE", "oblivious")
oblivious_mode = (classifier_mode == "oblivious")

if oblivious_mode:
    from StanceClassifier.stance_classifier import StanceClassifier
    args = {}
    if os.environ.get("STANCE_CLASSIFIER_OBLIVIOUS_MODEL", "default") != "default":
        args["model"] = os.environ["STANCE_CLASSIFIER_OBLIVIOUS_MODEL"]
    classifier = StanceClassifier(**args)
else:
    ensemble_mode = (classifier_mode == "ensemble")
    if ensemble_mode:
        from StanceClassifier.stance_classifier import StanceClassifierEnsemble
        args = {}
        if os.environ.get("STANCE_CLASSIFIER_OBLIVIOUS_MODEL", "default") != "default":
            args["to_model"] = os.environ["STANCE_CLASSIFIER_OBLIVIOUS_MODEL"]
        if os.environ.get("STANCE_CLASSIFIER_AWARE_MODEL", "default") != "default":
            args["ta_model"] = os.environ["STANCE_CLASSIFIER_AWARE_MODEL"]
        classifier = StanceClassifierEnsemble(**args)
    else:
        from StanceClassifier.stance_classifier import StanceClassifierWithTarget
        args = {}
        if os.environ.get("STANCE_CLASSIFIER_AWARE_MODEL", "default") != "default":
            args["model"] = os.environ["STANCE_CLASSIFIER_AWARE_MODEL"]
        classifier = StanceClassifierWithTarget(**args)
