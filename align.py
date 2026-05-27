import numpy as np
import gotoh

str1 = "string1"
str2 = "string2"

reference_str = "Why did you why did you paint your car Um well there were two reasons oh one is the obvious one that you know you’d be looking for a a dark coloured one but the other one was it just needed a paint if you go and get your finger and just run across the top of the car um where it’s still the grey it the paint it’s all faded and it’ll actually come off on your fingers and ah it looked dreadful but the paint served its purpose just fine it’s a rustproofing paint but um I'd sort of painted it and and touched it up several times in that colour but the grey does fade very quickly and it comes off on your fingers so I was gonna paint it anyway Yeah Um I didn’t go and buy paint I just looked around in the garage there was some Sandbank there which I bought maybe a couple of years previously to paint the Jayco Hawk campervan then changed my mind on the colours and I painted that green and so I thought that'll do it was lockdown no income stood down the car needed to be painted suited that plan and so I painted it and look it’s the car to me look look most people would consider a car to be something of prestige but for me it’s just another tool like any other tool in the garage like a hammer or a chainsaw um it just needs to be functional uh it’s not a status thing for me it just needs to do its job and to keep it uh free of rust I painted it in um Dulux uh what’s it called Metalshield Mhm And it actually comes up with a quite a reasonable um look when you paint it with a a nice roller good-quality roller it doesn’t look too bad from a distance What did Melanie think when you were painting it Ah well she’d seen me paint it many times before Yeah with Dulux Metalshield so you know here he goes again he's painting his car So made no difference in reality No no as the neighbours saw it here he goes again he’s painting his car that might seem odd but as I explained it’s just a tool Yeah uh and a tool that you obviously enjoy Yeah it’s done four hundred and sixty six thousand kilometres ah the gearbox i- was completely um shot it was really low on power the torque converter was um not working and um getting up hills it really struggles I’ve just spent a whole lot of money getting it fixed so it has a reconditioned uh gearbox now and it’s uh running like like new"
response = "why did you paint your car two reasons you'd be looking for a dark coloured one right across the top all faded served its purpose just fine touched it up several times does fade very quickly didnâ€™t go and buy paint changed my mind and lock down no income stood down suited that plan most people would consider a car to be prestige hammer or a chain saw not a status thing for me free of rust dulux what's it called metalshield good quality roller what did melanie thing here he goes again he's painting his care it'd does 466 thousand kilometres completely shot getting up hills bit of a struggle"

reference_str = "Why did you why did you paint your car Um well there were two reasons oh one is the obvious one that you know you’d be looking for a a dark coloured one"
response = "why did you paint your car two reasons you'd be looking for a dark coloured one right"


str1 = reference_str
str2 = response

# str1_array = np.array([list(str1)], dtype=np.byte)
# str2_array = np.array([list(str2)], dtype=np.byte)

gap_open=-1
gap_extend=-0.02


# str1_array = np.array([np.frombuffer(str1.encode("utf-8"), dtype=np.uint8)], dtype=np.int)
# str2_array = np.array([np.frombuffer(str2.encode("utf-8"), dtype=np.uint8)], dtype=np.int)


str1_array = np.expand_dims(np.frombuffer(str1.encode("utf-8"), dtype=np.uint8), axis=1).astype(np.int)
str2_array = np.expand_dims(np.frombuffer(str2.encode("utf-8"), dtype=np.uint8), axis=1).astype(np.int)

x = gotoh.msa(
    str1_array, 
    str2_array,
    gap_open=gap_open,
    gap_extend=gap_extend,
    visualize=1
)
x[x == -1] = ord('_')

string1_gap = "".join(map(chr, x[:,0]))


string2_gap = "".join(map(chr, x[:,1]))
print(string1_gap[:200])
print(string2_gap[:200])
breakpoint()