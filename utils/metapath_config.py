"""
Author: Tian Yuxuan
Date: 2025-08-04
"""
view_metas = {
    'patient_disease': (
        ['visit', 'co'],
        [
            ('co', 'in', 'visit'),
            ('visit', 'has', 'co'),
            ('visit', 'connect', 'visit'),
        ]
    ),
    'treatment_path': (
        ['visit', 'pr', 'dh'],
        [
            ('pr', 'in', 'visit'),
            ('dh', 'in', 'visit'),
            ('visit', 'has', 'pr'),
            ('visit', 'has', 'dh'),
            ('visit', 'connect', 'visit'),
        ]
    ),
    'drug_disease': (
        ['visit', 'co', 'dh'],
        [
            ('co', 'in', 'visit'),
            ('visit', 'has', 'co'),
            ('dh', 'in', 'visit'),
            ('visit', 'has', 'dh'),
            ('visit', 'connect', 'visit'),
        ]
    ),
    'procedures_disease': (
        ['visit', 'co', 'pr'],
        [
            ('co', 'in', 'visit'),
            ('visit', 'has', 'co'),
            ('pr', 'in', 'visit'),
            ('visit', 'has', 'pr'),
            ('visit', 'connect', 'visit'),
        ]
    )
}