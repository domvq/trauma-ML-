
# ============================================================
# AI EMS PROTOCOL SUPPORT
#
# Reference framework:
# NASEMSO National Model EMS Clinical Guidelines
# Version 3.0, March 2022
#
# IMPORTANT:
# These are educational decision-support prompts.
# Local/state EMS protocols and medical direction take
# precedence.
# ============================================================


PROTOCOL_SOURCE = (
    "NASEMSO National Model EMS Clinical Guidelines, Version 3.0"
)


PROTOCOL_NOTE = (
    "Educational prototype. Verify all recommendations against "
    "current state/local EMS protocols and medical direction."
)


PROTOCOLS = {

    "laceration": {

        "title": "Laceration / Open Wound",

        "considerations": [

            "Assess the wound and mechanism of injury.",

            "Assess severity and presence of external hemorrhage.",

            "Assess distal circulation, sensation, and motor function "
            "when appropriate.",

            "Monitor and reassess vital signs.",

            "Consider associated injuries based on the mechanism.",

            "Use hemorrhage-control measures consistent with "
            "applicable EMS protocol and provider scope."
        ]
    },


    "burn": {

        "title": "Burn Injury",

        "considerations": [

            "Assess burn mechanism and circumstances.",

            "Assess burn location, extent, and characteristics.",

            "Assess airway and breathing, particularly when "
            "inhalation injury is suspected.",

            "Monitor oxygenation and other relevant vital signs.",

            "Assess for associated traumatic injuries.",

            "Prevent further exposure and continue appropriate "
            "supportive care according to applicable protocol.",

            "Consider destination and transport requirements "
            "according to local trauma/burn-system guidance."
        ]
    },


    "bruise": {

        "title": "Contusion / Bruising",

        "considerations": [

            "Assess mechanism of injury.",

            "Assess the affected area for additional injury.",

            "Consider associated soft-tissue or musculoskeletal "
            "injury.",

            "Assess distal circulation, sensation, and motor "
            "function when appropriate.",

            "Monitor and reassess vital signs.",

            "Consider hidden or associated trauma based on "
            "mechanism and patient presentation."
        ]
    },


    "abrasion": {

        "title": "Abrasion",

        "considerations": [

            "Assess mechanism of injury.",

            "Assess wound characteristics and extent.",

            "Evaluate for associated injuries.",

            "Assess pain and relevant vital signs.",

            "Monitor for signs suggesting a more significant "
            "injury than the visible wound alone."
        ]
    },


    "normal skin": {

        "title": "No Obvious Wound Classification",

        "considerations": [

            "The computer-vision model did not identify one "
            "of the trained wound categories.",

            "Continue a complete patient assessment.",

            "Consider mechanism of injury and patient complaints.",

            "Do not use the image classification alone to exclude "
            "injury."
        ]
    }
}


def get_protocol(injury_class):

    key = injury_class.lower().strip()

    return PROTOCOLS.get(
        key,
        {
            "title": "General Assessment",
            "considerations": [
                "Continue appropriate patient assessment.",
                "Follow applicable local EMS protocol."
            ]
        }
    )

