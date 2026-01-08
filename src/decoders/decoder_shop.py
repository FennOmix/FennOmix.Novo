"""预留：未使用"""
from typing import Dict, Optional, Tuple, Type
from .base_decoder import DecoderHead, DecoderTail
from .peptide_decoder_head import PeptideDecoderHead
from .dp_decoder_tail import DPDecoderTail


class DecoderShop:
    """
    两层 Decoder 工厂
    【第一层】DecoderHead：生成概率分布
    【第二层】DecoderTail：搜索最优序列
    """
    # 第一层：生成层
    _head_decoders: Dict[str, Type[DecoderHead]] = {
        'peptide': PeptideDecoderHead,
    }

    # 第二层：搜索层
    _tail_decoders: Dict[str, Type[DecoderTail]] = {
        'dp': DPDecoderTail,
    }

    def __init__(self):
        self.head = None
        self.tail = None

    @classmethod
    def register_head(cls, name: str, head_class: Type[DecoderHead]) -> None:
        """注册新的 DecoderHead"""
        cls._head_decoders[name] = head_class
        print(f"✅ Registered DecoderHead: {name}")
    @classmethod
    def register_tail(cls, name: str, tail_class: Type[DecoderTail]) -> None:
        """注册新的 DecoderTail"""
        cls._tail_decoders[name] = tail_class
        print(f"✅ Registered DecoderTail: {name}")

    def create_head(self, name: str, **kwargs) -> DecoderHead:
        """创建 DecoderHead"""
        if name not in self._head_decoders:
            raise ValueError(f"Unknown DecoderHead: {name}")

        head_class = self._head_decoders[name]
        self.head = head_class(**kwargs)
        print(f"✅ Created DecoderHead: {name}")
        return self.head

    def create_tail(self, name: str, **kwargs) -> DecoderTail:
        """创建 DecoderTail"""
        if name not in self._tail_decoders:
            raise ValueError(f"Unknown DecoderTail: {name}")

        tail_class = self._tail_decoders[name]
        self.tail = tail_class(**kwargs)
        print(f"✅ Created DecoderTail: {name}")
        return self.tail

    def create(
            self,
            head_type: str = 'peptide',
            tail_type: str = 'dp',
            head_kwargs: Optional[Dict] = None,
            tail_kwargs: Optional[Dict] = None,
            ) -> Tuple[DecoderHead, DecoderTail]:
        """
        创建完整的两层解码器

        Example:
            shop = DecoderShop()
            head, tail = shop.create(
                head_type='peptide',
                tail_type='dp',
                head_kwargs={'dim_model': 512, 'n_layers': 9},
                tail_kwargs={'top_k': 10}
            )
        """
        head_kwargs = head_kwargs or {}
        tail_kwargs = tail_kwargs or {}

        head = self.create_head(head_type, **head_kwargs)
        tail = self.create_tail(tail_type, **tail_kwargs)

        return head, tail

    def list_heads(self) -> Dict[str, Type[DecoderHead]]:
        """列出所有可用的 DecoderHead"""
        return self._head_decoders.copy()
    def list_tails(self) -> Dict[str, Type[DecoderTail]]:
        """列出所有可用的 DecoderTail"""
        return self._tail_decoders.copy()
    def __repr__(self) -> str:
        return (
            f"DecoderShop(\n"
            f"  Heads: {list(self._head_decoders.keys())},\n"
            f"  Tails: {list(self._tail_decoders.keys())}\n"
            f")"
        )
__all__ = ['DecoderShop', 'DecoderHead', 'DecoderTail',
           'PeptideDecoderHead', 'DPDecoderTail']


