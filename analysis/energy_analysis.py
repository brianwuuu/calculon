# Utility function
def joulesPerBitToWatt(joules_per_bit, data_rate):
    return joules_per_bit * data_rate

##### GPUs #####
class GPU:
    def __init__(self, name):
        self.name = name
        self.gpu_power_table = {
            "A100": 400,
            "H100": 700
        }
        assert(self.name in self.gpu_power_table.keys())

    def getPower(self):
        return self.gpu_power_table[self.name]
        

##### Transceiver #####
class Transceiver:
    def __init__(self, type, xcvr_bw_gbps):
        self.xcvr_type = type
        self.xcvr_power_table = {
            "IB": 16.5 / 800 * xcvr_bw_gbps,
            "NV": joulesPerBitToWatt(1.3 * 10 ** (-12), xcvr_bw_gbps * 10 ** (9)), # 7.2 Tb/s
            "SiP": joulesPerBitToWatt(818 * 10 ** (-15), xcvr_bw_gbps * 10 ** (9)) # 1.024 Tb/s
        }
        self.xcvr_bw_gbps = xcvr_bw_gbps
        assert(self.xcvr_type in ["IB", "NV", "SiP"])
        
    def getPower(self):
        return self.xcvr_power_table[self.xcvr_type]

    def getType(self):
        return self.xcvr_type

##### Link #####  
class Link:
    def __init__(self, type, link_bw_gbps):
        self.link_type = type
        self.link_bw_gbps = link_bw_gbps
        assert(self.link_type in ["IB", "NV", "SiP"])
        self.xcvr = Transceiver(type=self.link_type, xcvr_bw_gbps=link_bw_gbps)
    
    def getPower(self):
        # SiP links only have one transceiver as the other end of the transceiver 
        # terminates in a WSS which does not require EO conversion
        # NVLinks are electrical: xcvr power = link power
        xcvr_power = self.xcvr.getPower()
        return xcvr_power * 2 if self.link_type in ["IB"] else xcvr_power

    def getType(self):
        return self.link_type

##### Switch #####
class Switch:
    def __init__(self, type, radix, port_bw_gbps, num_lambda=32):
        self.switch_type = type
        self.radix = radix
        self.switch_power_table = {
            "IB": 747 / (32 * 800) * radix * port_bw_gbps, # 600 W / 64 ports
            "NV": 100 / (18 * 400) * radix * port_bw_gbps, # 100 W / 18 ports
            "SiP": (radix + radix) * (num_lambda + 1) * 0.0152 / (18 * 1024) * radix * port_bw_gbps # joulesPerBitToWatt(52 * 10 ** (-15), 16 * 10 ** (9))
        }
        assert(self.switch_type in self.switch_power_table.keys())
        
    def getPower(self):
        return self.switch_power_table[self.switch_type]
    
    def getType(self):
        return self.switch_type

##### Network #####
class Network:
    def __init__(self, gpu_type="A100", network_type="IB", ng=1, nl=1, ns=1, switch_radix=18, link_bw_gbps=800, pue=1.1):
        self.gpu_type = gpu_type
        self.network_type = network_type
        self.num_gpus = ng
        self.num_links = nl
        self.num_switches = ns
        self.switch_radix = switch_radix
        self.link_bw_gbps = link_bw_gbps
        self.pue = pue
        self.buildNetwork()
    
    def buildNetwork(self):
        assert(self.network_type in ["IB", "NV", "SiP"])
        self.gpu = GPU(self.gpu_type)
        self.link = Link(type=self.network_type, link_bw_gbps=self.link_bw_gbps)
        self.switch = Switch(type=self.network_type, port_bw_gbps=self.link_bw_gbps, radix=self.switch_radix)
    
    def getTotalGPUPower(self):
        return self.gpu.getPower() * self.num_gpus

    def getTotalLinkPower(self):
        return self.link.getPower() * self.num_links
    
    def getTotalSwitchPower(self):
        return self.switch.getPower() * self.num_switches
        
    def getTotalPower(self):
        gpu_power = self.getTotalGPUPower()
        link_power = self.getTotalLinkPower()
        switch_power = self.getTotalSwitchPower()
        return (gpu_power + link_power + switch_power) * self.pue
        

def computeTotalPower(gpu_type="A100", ng=1, nl=1, ns=1, switch_radix=18, link_bw_gbps=800, pue=1.1):
    
    ib_network = Network(gpu_type=gpu_type, network_type="IB", ng=ng, nl=nl, ns=ns, switch_radix=switch_radix, link_bw_gbps=link_bw_gbps, pue=pue)
    nv_network = Network(gpu_type=gpu_type, network_type="NV", ng=ng, nl=nl, ns=ns, switch_radix=switch_radix, link_bw_gbps=link_bw_gbps, pue=pue)
    sip_network = Network(gpu_type=gpu_type, network_type="SiP", ng=ng, nl=nl, ns=ns, switch_radix=switch_radix, link_bw_gbps=link_bw_gbps, pue=pue)

    power_ib_total = ib_network.getTotalPower()
    power_nv_total = nv_network.getTotalPower()
    power_sip_total = sip_network.getTotalPower()

    power_stats = {"ib": power_ib_total, "nv": power_nv_total, "sip": power_sip_total}
    # print(power_ib_total, power_nv_total, power_sip_total)
    return power_stats

if __name__ == "__main__":
    computeTotalPower()