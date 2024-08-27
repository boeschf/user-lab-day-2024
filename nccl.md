---
marp: true
theme: cscs
# class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('slides-support/common/4k-slide-bg-white.png')
size: 16:9
---

![bg cover](slides-support/common/title-bg3.png)
<!-- _paginate: skip  -->
<!-- _class: titlecover -->
<!-- _footer: "" -->

<div class="twocolumns">
<div>

# **Userlab Day 2024**

### NCCL

#### CSCS


</div>
<div>

![height:900px ](qr_rb.png)

</div>
</div>

---

# NVIDIA Collective Communication Library (NCCL)

- inter-GPU (collective) communication primitives

|  Send/Recv   | Broadcast     | Reduce | AllReduce | AllGather | ReduceScatter |
|---|---|---|---|---|---|

- single kernel implementation for communication and computation
- topology-aware, supports NVLINK
- API is similar to MPI
- bootstrap for parallel environment is out-of-band (not provided)
- can be used in combination with MPI

---

# Communicators and Communication Groups

- communication primitives transfer data among members of a communication `group`
- each group member corresponds to a CUDA device index
- a `communicator` refers to a particular `group`



---

# Uenv and Containers: aws-ofi-plugin

---

# NCCL and Cray Slingshot


---

# Performance: NCCL vs MPI


---

# Links

- reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage.html
- nccl-tests: https://github.com/NVIDIA/nccl-tests
- pytorch-nccl-tests: https://github.com/stas00/ml-engineering/tree/master/network/benchmarks

## Github Repo with slides and code

- https://github.com/boeschf/user-lab-day-2024

![width:400px](qr.png)

