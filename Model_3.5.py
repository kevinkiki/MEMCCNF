import sys
import ast
import pandas as pd
import docplex.mp
from collections import namedtuple
from docplex.mp.model import Model
from docplex.mp.environment import Environment
import docplex.mp.conflict_refiner as cr
import random
import time

############ UTIL ############
conv_bin = lambda x: 0 if x==0 else 1

def str2list(x:str):
    return list(ast.literal_eval(x))

def route_df_to_tuple(pc,total_port,port_country):
    routes_ind = []
    cnt,ind = 0,0
    for ins in pc:
        pc = ins[0]
        route = []
        legs= ins[-1]
        cnt += legs
        wk = ins[-2]
        econ = []
        for l in range(legs):
            call_port = pc[l]
            try:
                call = total_port.index(call_port)
            except:
                print(pc,len(routes_ind),call_port)
                raise
            route.append(call)
            econ.append(port_country[call])
        econ = tuple(dict.fromkeys(econ))
        routes_ind.append((wk,legs,tuple(pc),tuple(route),econ))
        ind += 1
    return routes_ind

############ Constant ############
#### Economics ####
INT_COD = ('JPN',  'LKA',  'HKG',  'CHN',  'SGP',  'MYS',  'KOR',  'IND',  'OAS',  'PAK',  'IDN',  'PHL',  'BGD',  'THA',  'VNM',  'KHM',  'MMR')
INT_COUNTRY = {'BD': 'BGD','CN': 'CHN','HK': 'HKG','IN': 'IND','ID': 'IDN', 'JP': 'JPN',
  'KH': 'KHM','KR': 'KOR','MM': 'MMR', 'MY': 'MYS','PK': 'PAK','PH': 'PHL','SG': 'SGP',
   'LK': 'LKA',  'TW': 'OAS',  'TH': 'THA',  'VN': 'VNM' }
REGION_COD = ('AFR','EUR','ELT','WLT','MDE','NAM','OCE')
CON_INT = dict(zip(INT_COUNTRY.values(),INT_COUNTRY.keys()))

assert len(INT_COD ) == len(INT_COUNTRY)
assert len([i for i in INT_COUNTRY.values() if i not in INT_COD]) ==0
ECONOMICS = tuple(INT_COD + REGION_COD)
print(f"Total Economics {len(ECONOMICS)}")   
print(f"Interested Economics {len(CON_INT)}")   
print(f"External Economics {len(REGION_COD)}")  

#### Vessel types ####
VESSEL_TYPE = [
    ('Feeder',      3000,   8,   170, 1700, 146 ),
    ('Panamax',     8500,   11,  275, 1100, 220),
    ('Neo_Panamax', 14000,  13,  335, 600,  326),
    ('ULCV',        25000,  14,  390, 200,  506)]
# Cost in week base (k$)
Vessel_type = namedtuple('vessel_type',['type','teu','draft','length','quantity','deploy_cost'])
VTYPES = [Vessel_type(*v) for v in VESSEL_TYPE]
print(f"Total Vessel types {len(VTYPES)}")    #552

#### Demand ####
# year = input("Year Select (2012-2015) : ")
year = 2015
DEM_MATRIX = pd.read_csv(f'load_matrix_{year}.csv').iloc[:,1:]
DEM,TOT_DEM = {},{}
for f in ECONOMICS:
    TOT_DEM[f] = 0
    for t in ECONOMICS:
        if f==t : continue
        DEM[f,t] = DEM_MATRIX[(DEM_MATRIX.From==f)&(DEM_MATRIX.To==t)].TEU.sum()//50
        TOT_DEM[f] += DEM[f,t]
assert len(DEM) == len(ECONOMICS) *(len(ECONOMICS)-1)
ANNUAL_TEU = pd.read_csv(f'TEU_{year}.csv')
RDEM = dict(zip(ANNUAL_TEU.Econ,ANNUAL_TEU.TEU//50))
print(f"Total Demand {len(DEM)}")    #552

#### Port ####
master_port = pd.read_csv('ports_master.csv').iloc[:,1:]
PORTS = list(master_port[['LOCODE','draft0_r','draft1_r','draft2_r','draft3_r',
                        'due_0', 'due_1', 'due_2', 'due_3', 
                        'thc/teu','tranship/teu']].itertuples(index=False, name=None))
TOTAL_PORT = tuple(master_port.LOCODE.tolist() + list(REGION_COD))
for reg in REGION_COD:
    sc = (reg, 20000, 20000, 20000, 20000, 0, 0, 0, 0, 0, 999)
    PORTS.append(sc)
INTEREST_PORT = tuple(master_port.LOCODE.tolist())
Ports_info = namedtuple('ports_info',['lOCODE','draft0_r','draft1_r','draft2_r','draft3_r','due_0', 'due_1', 'due_2', 'due_3', 'thc','tranship'])
PORTS_INFO = [Ports_info(*p) for p in PORTS]
PORT_COUNTRY = {i:ECONOMICS.index(INT_COUNTRY[p[:2]]) for i,p in enumerate(TOTAL_PORT) if len(p)==5}
PORT_COUNTRY.update({i:ECONOMICS.index(p) for i,p in enumerate(TOTAL_PORT) if len(p)==3})
print(f"Total Ports {len(PORTS)}")    #552

#### Routes ####
ROUTES_DF = pd.read_csv('Port_call_master.csv').iloc[:,1:]
ROUTES_DF = ROUTES_DF[ROUTES_DF.Calls_Count>0].reset_index(drop=True)
ROUTES_DF.LOCODE_Calls = ROUTES_DF.LOCODE_Calls.apply(str2list)
ROUTES_DF.Calls = ROUTES_DF.Calls.apply(str2list)
PORTCALL = list(ROUTES_DF[['Calls','PC_Count','Area_Calls',
                        'Area_Cover','Turn_Around','Week','Calls_Count']].itertuples(index=False, name=None))
ROUTES = route_df_to_tuple(PORTCALL,TOTAL_PORT,PORT_COUNTRY)
print(f"Total routes {len(ROUTES)}")    #552

def model():
    global ECONOMICS,INT_COD,REGION_COD
    global TOTAL_PORT,ROUTES,VTYPES,PORTS_INFO,PORT_COUNTRY,DEM,TOT_DEM
    ############ Iterator ############
    # Economics
    _economics = range(len(ECONOMICS))

    #Ports
    _total_port     = range(len(TOTAL_PORT))

    # Routes
    _routes = range(len(ROUTES))

    # Vessel 
    _vtypes = range(len(VTYPES))

    # Cross Domain
    rv =    [(r,v)        for r in _routes for v in _vtypes] 
    rolp =  [(r,o,l,ROUTES[r][3][l])    for r in _routes for o in _economics 
                                            for l in range(ROUTES[r][1])]
    ope =   [(o,_p,e )    for o in _economics for _p,e in PORT_COUNTRY.items()]   
    pv =    [(p,v)        for p in _total_port for v in _vtypes]
    print("#"*45)
    print(' Created by Krittitee Yanpisitkul '.center(45,"#"))
    print("#"*45)
    ############ Model ############

    env = Environment()
    env.print_information()
    print("#"*45)
    print(f"Model {year} Initiate")
    print("#"*45)
    mdl = Model(f"Container_network_3.2_{year}")

    ############ Decision Variables ############
    ### Main DV ###
    # Vessel in route (Ves=>)
    ves_deploy = mdl.integer_var_dict(rv,  name="ves_deploy") # Maximum 7 day/weeks call

    # Load, Unload, and Carried
    load = mdl.continuous_var_dict(rolp,name="load")
    unload = mdl.continuous_var_dict(rolp, name="unload")
    carried = mdl.continuous_var_dict(rolp, name="carried")

    # Import,Export
    imp = mdl.continuous_var_dict(ope,name="imp")
    exp = mdl.continuous_var_dict(ope,name="exp")
    # trade = mdl.continuous_var_dict(ope,name="trade")
    total_var = len(rolp )*3 + len(ope)*2 + len(rv)
    print(f"Total DVs {total_var}")
    print("#"*45)

    ### Cal Var ###
    port_calls =  mdl.continuous_var_dict(pv,name="port_calls")
    route_cap =  mdl.continuous_var_list(_routes,name="route_cap")
    port_load =  mdl.continuous_var_dict(ope,name="port_load")
    port_unload = mdl.continuous_var_dict(ope,name="port_unload")
    port_trans  = mdl.continuous_var_dict(ope,name="port_trans")
    port_thrp =  mdl.continuous_var_list(_total_port,name="port_thrp")

    ############ Constraints ############
    ### C1 Capacity on each route
    for r in _routes:
        mdl.add_constraint(
                ## deploy * size 
                mdl.sum((ves_deploy[(r,v)]*VTYPES[v].teu) for v in _vtypes) 
                # + relax_route_cap[r]    # Relaxed
                == route_cap[r],f"Cap route {r}"
            ) 

    ### C2 Vessel Availability (Vessel deploy according to weekly service)
    for v in _vtypes:
        mdl.add_constraint(
                ## deploy * week
                mdl.sum((ves_deploy[(r,v)]*ROUTES[r][0]) for r in _routes)
                <= VTYPES[v].quantity,f"Availability type {v}")

    ### C3 Port Total Call (by types)
        for _p,p in enumerate(TOTAL_PORT): 
            mdl.add_constraint(mdl.sum(ves_deploy[(r,v)]/24 
                for r,o,l,p in rolp if p==_p and l< ROUTES[r][1]-1) # Use 1 to reduce end duplicate 
                ==port_calls[(_p,v)],f"Total Call port {p}, type {v}")

    ### C4 Route flow conservation on each leg
    ### C5 Carried less than vessel capacity
    for r in _routes:
        for o in _economics:
            mdl.add_constraint( mdl.sum((load[(r,o,l,p)] - unload[(r,o,l,p)]) 
            for l in range(ROUTES[r][1]-1) for p in [ROUTES[r][3][l]])
            == 0,
            f"Route flow conservation {r}{o}"  )

        for l in range(ROUTES[r][1]):
            p = ROUTES[r][3][l]
            mdl.add_constraint( mdl.sum(carried[(r,o,l,p)] 
                for o in _economics)
                <= route_cap[r],f"Cap & Carried  {r},{l}")

    ### C6 Port Flow (vessel) Continuity (per each berthing) 
    ### C7 Circular Route (PC[0] ==PC[-2])
    for _rol in rolp :
        r,o,l,p = _rol
        # No unload at origin (no import @ origin)
        if PORT_COUNTRY[p]==o:
            mdl.add_constraint(unload[_rol]==0,f"No origin unload {p},{o}")
            
        # first == last portcall since is identical with first portcall see C5
        if l==ROUTES[r][1]-1:    
            # print(r,l,"  x") 
            p0 = ROUTES[r][3][0]
            mdl.add_constraint(carried[_rol]== carried[(r,o,0,p0)] ,f"Circ Carried {_rol}")
            mdl.add_constraint(load[_rol]   == load[(r,o,0,p0)]   , f"Circ Load {_rol}")
            mdl.add_constraint(unload[_rol] == unload[(r,o,0,p0)]   ,f"Circ unload {_rol}")
        elif l<ROUTES[r][1]-1 :
            # print(r,l,)
            _p = ROUTES[r][3][l+1]
            mdl.add_constraint(carried[(r,o,l+1,_p)] == carried[_rol] + load[_rol] - unload[_rol],f"R Continuity {_rol}")

    ### C8 Port Load, Unload, No internal trade, & Throughput
    # TH = load + unload
    # For port in e can only load cargoes from e

    for _o,_p,_e in ope:
        # Convervation of flow @ port
        mdl.add_constraint( mdl.sum((load[(r,o,l,p)] - unload[(r,o,l,p)] )
                                for r,o,l,p in rolp 
                                if p==_p and o==_o and l<ROUTES[r][1]-1) 
                            - exp[(_o,_p,_e)] + imp[(_o,_p,_e)]
                            == 0,
                            f"Port Flow conservation {_p},{_o},{_e}")

        # Port load
        mdl.add_constraint( mdl.sum(load[(r,o,l,p)] 
                                for r,o,l,p in rolp 
                                if p==_p and o==_o and l<ROUTES[r][1]-1) 
                            == port_load[(_o,_p,_e)],
                            f"Sum Port Load {_p},{_o},{_e}")

        # Port unload
        mdl.add_constraint( mdl.sum(unload[(r,_o,l,p)]  
                            for r,o,l,p in rolp if p==_p and o==_o and l<ROUTES[r][1]-1) 
                            == port_unload[(_o,_p,_e)],
                            f"Sum Port Unload {_p},{_o},{_e}") 
                            
        if _o == _e:
            # Port unload product 
            mdl.add_constraint( port_unload[(_o,_p,_e)] == 0,
                                    f"Port Unload Fix=0 {_p},{_o}")
        else:
            mdl.add_constraint( port_unload[(_o,_p,_e)] + port_load[(_o,_p,_e)] - imp[(_o,_p,_e)]
                                ==2*port_trans[(_o,_p,_e)],
                                f"Port_trans_{p}_{_p}_{_e}")

    for _p,p in enumerate(TOTAL_PORT):
        _e = PORT_COUNTRY[_p]
        # Port Throughput
        mdl.add_constraint( mdl.sum(port_unload[(_o,_p,_e)] for _o in _economics)
                            + mdl.sum(port_load[(_o,_p,_e)] for _o in _economics)
                            ==port_thrp[_p],f"Port Throughput {p}")

    ### C9 Berth Draft Limitation
    for v in _vtypes:
        _v = len(VTYPES) - v -1
        for _p,p in enumerate(TOTAL_PORT) :
            mdl.add_constraint(sum([PORTS[_p][1+v_]*7 for v_ in range(_v,len(VTYPES))])
                            >= mdl.sum([port_calls[(_p,v_)] * VTYPES[v_].length for v_ in range(_v,len(VTYPES))])
                            ,f"Berth Limitation port {p} type {v}")

    ### C10 Trade for each country
    ### C11 Demand Satisfied (Only in interested Area)
    ### C12 No internal trade

    for _f,f in enumerate(ECONOMICS):          # @ Economy f
        fports = [(_p,TOTAL_PORT[_p]) for _p,c in PORT_COUNTRY.items() if c == _f]
        # tports = [(_p,total_port[_p]) for _p,c in port_country.items() if c == _t]
        if f in INT_COD:
            # export from interest port
            mdl.add_constraint(mdl.sum(exp[(_e,_p,_f)] 
                                for _p,p in fports 
                                for _e in _economics)
                            ==TOT_DEM[f],
                            f"export_intreg_{f}")
        else:
            # export from ex region
            mdl.add_constraint(mdl.sum(exp[(_e,_p,_f)] 
                                for _p,p in fports 
                                for _e in _economics)
                            <=TOT_DEM[f],
                            f"export_extreg_{f}")

        for _t,t in enumerate(ECONOMICS):       # Partner
            if f==t:
                # Import of (product form econ t) from econ t is 0
                mdl.add_constraints([imp[(_t,_p,_f)]==0 for _p,p in fports],
                                    [f"import_none_{p} for {f}" for _p,p in fports]) 
            else:
                # Export of (product form econ t) from econ not t is 0 
                mdl.add_constraints([exp[(_t,_p,_f)]==0 for _p,p in fports],
                                    [f"export_none_{p} for {f}" for _p,p in fports])
                if f in INT_COD:
                    # import from t to f (interested area)
                    mdl.add_constraint(mdl.sum(imp[(_t,_p,_f)] 
                                    for _p,p in fports)==DEM[t,f],
                                    f"demands_satisfy_{t,f}")
                else:
                    # import from t to f (outside area)
                    mdl.add_constraint(mdl.sum(imp[(_t,_p,_f)] 
                                    for _p,p in fports)<=DEM[t,f],
                                    f"demands_relaxed_{t,f}")

    ############ Cost Calculation ############
    # Vessel deploy cost 
    ves_deploy_cost = mdl.sum(ves_deploy[(r,v)] * VTYPES[v].deploy_cost for r,v in rv)
    mdl.add_kpi(ves_deploy_cost, "ves_deploy_cost")
    # Port Call cost
    port_call_cost = mdl.sum(ves_deploy[(r,v)] * PORTS_INFO[_p][v+5] for r,v in rv
                                for l in range(ROUTES[r][1]-1) 
                                for _p in [ROUTES[r][3][l]])
    mdl.add_kpi(port_call_cost, "  Port_Call_cost")
    # Load & unload cost

    # ** Adjust cost for external 
    handling_cost = mdl.sum(port_thrp[_p] * PORTS_INFO[_p][-2] for _p in _total_port)
    mdl.add_kpi(handling_cost, "Handling_cost")

    # Transhipment cost
    trans_cost = mdl.sum(port_trans[(o,_p,e)] * PORTS_INFO[_p][-1] for o,_p,e in ope)
    mdl.add_kpi(trans_cost, "Transshipment_cost")

    mdl.minimize(ves_deploy_cost+ port_call_cost +trans_cost +handling_cost)
    mdl.print_information()
    s = mdl.solve(log_output=True)
    if s is not None:
        mdl.report()
    else:
        print("* model is infeasible")
    tref = time.strftime("%m%d%H%M",time.localtime())
    mdl.export_as_lp(path='.')
    with open("solve_log.txt",'a') as f:
        f.write(mdl.solve_details.status())
        f.write(tref)
    time_solve = mdl.solve_details.time
    mdl.solution.export(f"Solution_{year}_{tref}.json")
    
    return tref,time_solve

if __name__ == "__main__":
    for _ in range(1):
        tr = model()
        print(tr) 