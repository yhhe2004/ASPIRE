from typing import Iterable

import torchdata.datapipes as dp

import freerec
from freerec.data.postprocessing.sampler import (
    NUM_NEGS_FOR_SAMPLE_BASED_RANKING,
    ValidSampler,
)


@dp.functional_datapipe("aspire_valid_sampling_")
class ASPIREValidSampler(ValidSampler):
    def __init__(
        self,
        source,
        ranking="full",
        num_negatives=NUM_NEGS_FOR_SAMPLE_BASED_RANKING,
        val_data=None,
    ):
        self.val_data = val_data
        super().__init__(source, ranking, num_negatives)

    def _build_val_holdout(self) -> tuple[set[int], ...]:
        holdout = [set() for _ in range(self.User.count)]

        if self.val_data is None:
            return tuple(holdout)

        for user, item in zip(self.val_data[0].tolist(), self.val_data[1].tolist()):
            holdout[user].add(item)
        return tuple(holdout)

    @freerec.utils.timemeter
    def prepare(self):
        seenItems = [[] for _ in range(self.User.count)]
        unseenItems = [[] for _ in range(self.User.count)]
        val_holdout = self._build_val_holdout()

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.train().to_seqs(),
        )

        self.listmap(
            lambda row: unseenItems[row[self.User]].extend(
                self._filter_val_items(row[self.ISeq], val_holdout[row[self.User]])
            ),
            self.dataset.valid().to_seqs(),
        )

        self.seenItems = seenItems
        self.unseenItems = unseenItems
        self.negItems = dict()

    @staticmethod
    def _filter_val_items(items: Iterable[int], holdout: set[int]) -> list[int]:
        return [item for item in items if item not in holdout]
