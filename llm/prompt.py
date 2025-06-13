from textwrap import dedent

SYS_TEMPLATE = dedent("""\
You are an expert molecular-design assistant.
Current molecule (SMILES): {smiles}
Generate **one** modified molecule for the given action.
Return ONLY a JSON list of SMILES strings.
""")

ACTION_TEMPLATE = "\nAction: {type} with params {params}"