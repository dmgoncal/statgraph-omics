import numpy as np


TARGETS = ["vital_status", "primary_diagnosis"]

PROJECT_CONFIG = {
    "TCGA-LGG": {
        "vital_status": {
            "drop": ["Not Reported", np.nan],
            "replace": {}
        },
        "primary_diagnosis": {
            "drop": [np.nan],
            "replace": {
                'Astrocytoma, anaplastic': 'Astrocytoma',
                'Astrocytoma, NOS': 'Astrocytoma',
                'Oligodendroglioma, anaplastic': 'Oligodendroglioma',
                'Oligodendroglioma, NOS': 'Oligodendroglioma'
            }
        }
    },
    "TCGA-COAD": {
        "vital_status": {
            "drop": [np.nan],
            "replace": {}
        },
        "primary_diagnosis": {
            "drop": [
                np.nan,
                'Papillary adenocarcinoma, NOS',
                'Adenocarcinoma with neuroendocrine differentiation',
                'Adenocarcinoma with mixed subtypes',
                'Carcinoma, NOS',
                'Adenosquamous carcinoma'
            ],
            "replace": {}
        }
    },
    "TCGA-KIRC": {
        "vital_status": {
            "drop": [],
            "replace": {}
        },
        "primary_diagnosis": {
            "drop": [],
            "replace": {}
        }
    },
    "TCGA-LUAD": {
        "vital_status": {
            "drop": [np.nan],
            "replace": {}
        },
        "primary_diagnosis": {
            "drop": [
                'NaN',
                'Signet ring cell carcinoma',
                'Clear cell adenocarcinoma, NOS',
                'Bronchiolo-alveolar adenocarcinoma, NOS',
                'Micropapillary carcinoma, NOS',
                'Bronchio-alveolar carcinoma, mucinous',
                'Solid carcinoma, NOS'
            ],
            "replace": {}
        }
    },
    "TCGA-OV": {
        "vital_status": {
            "drop": [np.nan, 'Not Reported'],
            "replace": {}
        },
        # primary_diagnosis not used for OV
    }
}
