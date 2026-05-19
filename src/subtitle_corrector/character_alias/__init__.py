"""角色别名与配队简称模块。

加载人工审核的角色别名词典，区分 character_nickname 和 team_comp_slot_alias。
"""
from subtitle_corrector.character_alias.lexicon import CharacterAliasLexicon
from subtitle_corrector.character_alias.team_comp import TeamCompParser

__all__ = ["CharacterAliasLexicon", "TeamCompParser"]
