from subtitle_corrector.detector.base import Detector, DetectorPlugin
from subtitle_corrector.detector.entity import EntityConsistencyDetector
from subtitle_corrector.detector.jieba_span import JiebaSpanDetector
from subtitle_corrector.detector.language_model import LanguageModelDetector
from subtitle_corrector.detector.pinyin import PinyinAnomalyDetector
from subtitle_corrector.detector.terminology import TerminologyAnomalyDetector

__all__ = [
    "Detector",
    "DetectorPlugin",
    "EntityConsistencyDetector",
    "JiebaSpanDetector",
    "LanguageModelDetector",
    "PinyinAnomalyDetector",
    "TerminologyAnomalyDetector",
]
