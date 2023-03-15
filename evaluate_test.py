import evaluate 

module = evaluate.load("dvitel/codebleu")
src = 'class AcidicSwampOoze(MinionCard):§    def __init__(self):§        super().__init__("Acidic Swamp Ooze", 2, CHARACTER_CLASS.ALL, CARD_RARITY.COMMON, battlecry=Battlecry(Destroy(), WeaponSelector(EnemyPlayer())))§§    def create_minion(self, player):§        return Minion(3, 2)§'
tgt = 'class AcidSwampOoze(MinionCard):§    def __init__(self):§        super().__init__("Acidic Swamp Ooze", 2, CHARACTER_CLASS.ALL, CARD_RARITY.COMMON, battlecry=Battlecry(Destroy(), WeaponSelector(EnemyPlayer())))§§    def create_minion(self, player):§        return Minion(3, 2)§'
src = src.replace("§","\n")
tgt = tgt.replace("§","\n")
res = module.compute(predictions = [tgt], references = [[src]])
print(res)
#{'CodeBLEU': 0.9473264567644872, 'ngram_match_score': 0.8915993127600096, 'weighted_ngram_match_score': 0.8977065142979394, 'syntax_match_score': 1.0, 'dataflow_match_score': 1.0}
