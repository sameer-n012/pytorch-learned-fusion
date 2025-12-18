# Points of interest:
### graph drawer (not firing right now):
- draw_buffers() https://github.com/sameer-n012/pytorch-learned-fusion/blob/main/torch/_inductor/debug.py#L67
	- draw_graph() https://github.com/sameer-n012/pytorch-learned-fusion/blob/main/torch/_functorch/partitioners.py#L2977
		- _get_node_label() https://github.com/sameer-n012/pytorch-learned-fusion/blob/main/torch/fx/passes/graph_drawer.py#L226
### text graph printer:
- log_ir_post_fusion() call https://github.com/sameer-n012/pytorch-learned-fusion/blob/main/torch/_inductor/scheduler.py#L2757
	- ir_pre_fusion()/ir_post_fusion() https://github.com/sameer-n012/pytorch-learned-fusion/blob/main/torch/_inductor/debug.py#L584
		- _write_ir() https://github.com/sameer-n012/pytorch-learned-fusion/blob/main/torch/_inductor/debug.py#L593
			- debug_str_extra() https://github.com/sameer-n012/pytorch-learned-fusion/blob/main/torch/_inductor/scheduler.py#L1581
- fx_graph_transformed() call https://github.com/sameer-n012/pytorch-learned-fusion/blob/main/torch/_inductor/compile_fx.py#L1308

### fusion
- https://github.com/sameer-n012/pytorch-learned-fusion/blob/main/torch/_inductor/scheduler.py#L3966

# Example IR Debug String:
```
op243: SchedulerNode(ComputedBuffer)
op243.writes = [MemoryDep('buf243', c0, {c0: 8})]
op243.unmet_dependencies = [
    MemoryDep('buf238', c0, {c0: 6144}),
    MemoryDep('buf241', c0, {c0: 6144})
]
op243.met_dependencies = []
op243.outputs = [
    buf243: ComputedBuffer
    buf243.layout = FixedLayout('cpu', torch.float32, size=[1, 8, 1], stride=[8, 1, 8])
    buf243.users = [
        NodeUser(node=SchedulerNode(name='op245'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op249'), can_inplace=True, is_weak=False),
    ]
]
op243.group.device = cpu
op243.group.iteration = ((8,), (768,))
op243.sizes = ([8], [768])
buf241_layout = FixedLayout('cpu', torch.float32, size=[8, 768], stride=[768, 1])
buf238_layout = FixedLayout('cpu', torch.float32, size=[1, 8, 768], stride=[6144, 768, 1])
buf243_layout = FixedLayout('cpu', torch.float32, size=[1, 8, 1], stride=[8, 1, 8])
class op243_loop_body:
    var_ranges = {p0: 8, p1: 768}
    index0 = 768*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf241', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('buf238', get_index_1)
        add = ops.add(load, load_1)
        reduction = ops.reduction(torch.float32, torch.float32, 'welford_reduce', add)
        getitem = reduction[0]
        getitem_1 = reduction[1]
        getitem_2 = reduction[2]
        get_index_2 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf243', get_index_2, getitem_1)
        return store_reduction
op242_op243_op245_op246_op249.snodes[2] =
```
- Too much information?
    - want users/deps to look like `[op123, op456, ...]`
    - want opcode instead of loop body
    - don't care about unmet vs met dependencies?
    - should include layout
    -

- fusion score looks like `op14_op15, op17_op18: (2, False, 49216, -3)`
    - `node1, node2: (template_score, node1.is_reduction() == node2.is_reduction() and memory_score > 0, memory_score, proximity_score)`
    - higher is better, sorted in tuple order

Formatting AWS disk:
```
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /mnt/big
sudo mount /dev/nvme1n1 /mnt/big
sudo chmod -R 777 /mnt/big
sudo mkdir /mnt/big/huggingface
df -h
```
