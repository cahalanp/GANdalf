#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -5.86*x up 6.05*y
  direction 1.00*z
  location <0,0,50.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7}
#declare pale = finish {ambient 0.5 diffuse 0.85 roughness 0.001 specular 0.200 }
#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 specular 0.5 }
#declare jmol = finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.05 diffuse 0.3 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.01 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.070;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

cylinder {<-69.02, -37.57, -24.87>, < 23.40, -75.72, -26.30>, Rcell pigment {Black}}
cylinder {<-60.81, -14.04, -121.72>, < 31.62, -52.20, -123.15>, Rcell pigment {Black}}
cylinder {<-23.52,  75.35, -96.85>, < 68.90,  37.20, -98.27>, Rcell pigment {Black}}
cylinder {<-31.74,  51.83,   0.00>, < 60.69,  13.68,  -1.43>, Rcell pigment {Black}}
cylinder {<-69.02, -37.57, -24.87>, <-60.81, -14.04, -121.72>, Rcell pigment {Black}}
cylinder {< 23.40, -75.72, -26.30>, < 31.62, -52.20, -123.15>, Rcell pigment {Black}}
cylinder {< 60.69,  13.68,  -1.43>, < 68.90,  37.20, -98.27>, Rcell pigment {Black}}
cylinder {<-31.74,  51.83,   0.00>, <-23.52,  75.35, -96.85>, Rcell pigment {Black}}
cylinder {<-69.02, -37.57, -24.87>, <-31.74,  51.83,   0.00>, Rcell pigment {Black}}
cylinder {< 23.40, -75.72, -26.30>, < 60.69,  13.68,  -1.43>, Rcell pigment {Black}}
cylinder {< 31.62, -52.20, -123.15>, < 68.90,  37.20, -98.27>, Rcell pigment {Black}}
cylinder {<-60.81, -14.04, -121.72>, <-23.52,  75.35, -96.85>, Rcell pigment {Black}}
atom(< -1.22,  -0.41, -60.84>, 0.44, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #0
atom(< -1.34,   0.46, -61.86>, 0.44, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #1
atom(< -0.26,   0.52, -62.77>, 0.44, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #2
atom(<  0.89,  -0.24, -62.65>, 0.44, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #3
atom(<  1.09,  -0.88, -61.47>, 0.44, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #4
atom(<  0.04,  -0.99, -60.59>, 0.44, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #5
atom(< -2.08,  -0.55, -60.17>, 0.18, rgb <1.00, 1.00, 1.00>, 0.0, ase2) // #6
atom(< -2.28,   0.98, -62.11>, 0.18, rgb <1.00, 1.00, 1.00>, 0.0, ase2) // #7
atom(< -0.51,   1.30, -63.49>, 0.18, rgb <1.00, 1.00, 1.00>, 0.0, ase2) // #8
atom(<  1.61,  -0.21, -63.39>, 0.18, rgb <1.00, 1.00, 1.00>, 0.0, ase2) // #9
atom(<  2.09,  -1.33, -61.27>, 0.18, rgb <1.00, 1.00, 1.00>, 0.0, ase2) // #10
atom(<  0.16,  -1.47, -59.64>, 0.18, rgb <1.00, 1.00, 1.00>, 0.0, ase2) // #11
cylinder {< -1.22,  -0.41, -60.84>, < -1.28,   0.02, -61.35>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {< -1.34,   0.46, -61.86>, < -1.28,   0.02, -61.35>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {< -1.22,  -0.41, -60.84>, < -0.59,  -0.70, -60.71>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {<  0.04,  -0.99, -60.59>, < -0.59,  -0.70, -60.71>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {< -1.22,  -0.41, -60.84>, < -1.65,  -0.48, -60.51>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {< -2.08,  -0.55, -60.17>, < -1.65,  -0.48, -60.51>, Rbond texture{pigment {color rgb <1.00, 1.00, 1.00> transmit 0.0} finish{ase2}}}
cylinder {< -1.34,   0.46, -61.86>, < -0.80,   0.49, -62.32>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {< -0.26,   0.52, -62.77>, < -0.80,   0.49, -62.32>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {< -1.34,   0.46, -61.86>, < -1.81,   0.72, -61.99>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {< -2.28,   0.98, -62.11>, < -1.81,   0.72, -61.99>, Rbond texture{pigment {color rgb <1.00, 1.00, 1.00> transmit 0.0} finish{ase2}}}
cylinder {< -0.26,   0.52, -62.77>, <  0.32,   0.14, -62.71>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {<  0.89,  -0.24, -62.65>, <  0.32,   0.14, -62.71>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {< -0.26,   0.52, -62.77>, < -0.39,   0.91, -63.13>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {< -0.51,   1.30, -63.49>, < -0.39,   0.91, -63.13>, Rbond texture{pigment {color rgb <1.00, 1.00, 1.00> transmit 0.0} finish{ase2}}}
cylinder {<  0.89,  -0.24, -62.65>, <  0.99,  -0.56, -62.06>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {<  1.09,  -0.88, -61.47>, <  0.99,  -0.56, -62.06>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {<  0.89,  -0.24, -62.65>, <  1.25,  -0.22, -63.02>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {<  1.61,  -0.21, -63.39>, <  1.25,  -0.22, -63.02>, Rbond texture{pigment {color rgb <1.00, 1.00, 1.00> transmit 0.0} finish{ase2}}}
cylinder {<  1.09,  -0.88, -61.47>, <  0.57,  -0.93, -61.03>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {<  0.04,  -0.99, -60.59>, <  0.57,  -0.93, -61.03>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {<  1.09,  -0.88, -61.47>, <  1.59,  -1.10, -61.37>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {<  2.09,  -1.33, -61.27>, <  1.59,  -1.10, -61.37>, Rbond texture{pigment {color rgb <1.00, 1.00, 1.00> transmit 0.0} finish{ase2}}}
cylinder {<  0.04,  -0.99, -60.59>, <  0.10,  -1.23, -60.12>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{ase2}}}
cylinder {<  0.16,  -1.47, -59.64>, <  0.10,  -1.23, -60.12>, Rbond texture{pigment {color rgb <1.00, 1.00, 1.00> transmit 0.0} finish{ase2}}}
// no constraints
