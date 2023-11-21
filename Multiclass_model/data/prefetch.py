import torch


class PrefetchLoader:
    def __init__(
        self,
        loader,
        fp16=False,
    ):
        self.loader = loader
        self.fp16 = fp16

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                if isinstance(next_input, torch.Tensor):
                    next_input = next_input.cuda(non_blocking=True)
                    if self.fp16:
                        next_input = next_input.half()
                    else:
                        next_input = next_input.float()
                else:
                    if self.fp16:
                        next_input = [ni.cuda(non_blocking=True).half() for ni in next_input]
                    else:
                        next_input = [ni.cuda(non_blocking=True).float() for ni in next_input]
                if isinstance(next_target, torch.Tensor):
                    next_target = next_target.cuda(non_blocking=True)
                # elif isinstance(next_target, list):
                #     next_target = [nt.cuda(non_blocking=True) for nt in next_target]
                # Embs
                elif isinstance(next_target, list):
                    next_target_new = [nt.cuda(non_blocking=True) for nt in next_target[:-1]]
                    next_target_new.append(next_target[-1])
            if not first:
                yield input, target  # noqa: F821, F823
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
            # target = next_target_new

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
