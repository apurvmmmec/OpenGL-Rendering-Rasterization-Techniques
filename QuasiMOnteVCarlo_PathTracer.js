#define LIGHT
#define BOUNCE
#define THROUGHPUT
#define HALTON
#define IMPORTANCE_SAMPLING
#define AA


precision highp float;

#define M_PI 3.1415

struct Material {
  vec3 diffuse;
  vec3 specular;
  float glossiness;
  vec3 emission;
};

struct Sphere {
  vec3 position;
  float radius;
  Material material;
};

struct Plane {
  vec3 normal;
  float d;
  Material material;
};

const int sphereCount = 2;
const int planeCount = 2;
const int maxPathLength = 3;

struct Scene {
  Sphere[sphereCount] spheres;
  Plane[planeCount] planes;
};

struct Ray {
  vec3 origin;
  vec3 direction;
};

// Contains all information pertaining to a ray/object intersection
struct HitInfo {
  bool hit;
  float t;
  vec3 position;
  vec3 normal;
  Material material;
};

HitInfo getEmptyHit() {
  Material emptyMaterial;
  emptyMaterial.diffuse = vec3(0.0);
  emptyMaterial.specular = vec3(0.0);
  emptyMaterial.glossiness = 0.0;
  return HitInfo(false, 0.0, vec3(0.0), vec3(0.0), emptyMaterial);
}

HitInfo intersectSphere(Ray ray, Sphere sphere) {
    vec3 to_sphere = ray.origin - sphere.position;

    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(ray.direction, to_sphere);
    float c = dot(to_sphere, to_sphere) - sphere.radius * sphere.radius;
    float D = b * b - 4.0 * a * c;
    if (D > 0.0)
    {
    float t0 = (-b - sqrt(D)) / (2.0 * a);
    float t1 = (-b + sqrt(D)) / (2.0 * a);
        float t = min(t0, t1);
        vec3 hitPosition = ray.origin + t * ray.direction;
        return HitInfo(
            true,
            t,
            hitPosition,
            normalize(hitPosition - sphere.position),
            sphere.material);
    }
    return getEmptyHit();
}

HitInfo intersectPlane(Ray ray, Plane plane) {
  float t = -(dot(ray.origin, plane.normal) + plane.d) / dot(ray.direction, plane.normal);
  vec3 hitPosition = ray.origin + t * ray.direction;
  return HitInfo(
  true,
  t,
  hitPosition,
  normalize(plane.normal),
  plane.material); 
    return getEmptyHit();
}

float lengthSquared(vec3 x) {
  return dot(x, x);
}

HitInfo intersectScene(Scene scene, Ray ray, float tMin, float tMax)
{
    HitInfo best_hit_info;
    best_hit_info.t = tMax;
    best_hit_info.hit = false;

    for (int i = 0; i < planeCount; ++i) {
        Plane plane = scene.planes[i];
        HitInfo hit_info = intersectPlane(ray, plane);

        if( hit_info.hit && 
            hit_info.t < best_hit_info.t &&
            hit_info.t > tMin)
        {
            best_hit_info = hit_info;
        }
    }

    for (int i = 0; i < sphereCount; ++i) {
        Sphere sphere = scene.spheres[i];
        HitInfo hit_info = intersectSphere(ray, sphere);

        if( hit_info.hit && 
            hit_info.t < best_hit_info.t &&
            hit_info.t > tMin)
        {
            best_hit_info = hit_info;
        }
    }
  
  return best_hit_info;
}

// Converts a random integer in 15 bits to a float in (0, 1)
float randomIntegerToRandomFloat(int i) {
  return float(i) / 32768.0;
}

// Returns a random integer for every pixel and dimension that remains the same in all iterations
int pixelIntegerSeed(int dimensionIndex) {
  vec3 p = vec3(gl_FragCoord.xy, dimensionIndex);
  vec3 r = vec3(23.14069263277926, 2.665144142690225,7.358926345 );
  return int(32768.0 * fract(cos(dot(p,r)) * 123456.0));  
}

// Returns a random float for every pixel that remains the same in all iterations
float pixelSeed(int dimensionIndex) {
    return randomIntegerToRandomFloat(pixelIntegerSeed(dimensionIndex));
}

// The global random seed of this iteration
// It will be set to a new random value in each step
uniform int globalSeed;
int randomSeed;
void initRandomSequence() {
  randomSeed = globalSeed + pixelIntegerSeed(0);
}

// Computes integer x modulo y, not available in most WEBGL SL implementations
int mod(int x, int y) {
  return int(float(x) - floor(float(x) / float(y)) * float(y));
}

// Returns the next integer in a pseudo-random sequence
int rand() {
    randomSeed = randomSeed * 1103515245 + 12345;   
  return mod(randomSeed / 65536, 32768);
}

// Returns the next float in this pixels pseudo-random sequence
float uniformRandom() {
  return randomIntegerToRandomFloat(rand());
}

// Returns the ith prime number for the first 20 
const int maxDimensionCount = 10;
int prime(int index) {
  if(index == 0) return 2;
  if(index == 1) return 3;
  if(index == 2) return 5;
  if(index == 3) return 7;
  if(index == 4) return 11;
  if(index == 5) return 13;
  if(index == 6) return 17;
  if(index == 7) return 19;
  if(index == 8) return 23;
  if(index == 9) return 29;
  if(index == 10) return 31;
  if(index == 11) return 37;
  if(index == 12) return 41;
  if(index == 13) return 43;
  if(index == 14) return 47;
  if(index == 15) return 53;
  return 2;
}

//Code for generating Halton Sequence numbers. 
//This is used for deterministic sampling instead of random sampling
float halton(int sampleIndex, int dimensionIndex) {
#ifdef HALTON  
  // Code to generate Halton sequence
  float base = float(dimensionIndex);
  float n = float(sampleIndex);
    
  float haltNo = 0.0;
  float f = 1.0;
  for(int a=0;a<100;a++)
  {
    if(n<=0.0){
        return haltNo;
    }
    f=f/base;
    haltNo =haltNo+ f*float(mod(n,base));
    n=n/base;
  }  
  
#else
  return 0.0;
#endif
}

// This is the index of the sample controlled by the framework.
// It increments by one in every call of this shader
uniform int baseSampleIndex;

// Returns a well-distributed number in (0,1) for the dimension dimensionIndex
float sample(int dimensionIndex) {
#ifdef HALTON  
]  int pDIndex= prime(dimensionIndex);
  return fract(halton(baseSampleIndex,pDIndex)+pixelSeed(dimensionIndex));
#else
  return uniformRandom();
#endif  
}

// This is a helper function to sample two-dimensionaly in dimension dimensionIndex
vec2 sample2(int dimensionIndex) {
  return vec2(sample(dimensionIndex + 0), sample(dimensionIndex + 1));
}

// This is a register of all dimensions that we will want to sample.
//
// So if we want to use lens sampling, we call sample(LENS_SAMPLE_DIMENSION).
//
// There are infinitely many path sampling dimensions.
// These start at PATH_SAMPLE_DIMENSION.
// The 2D sample pair for vertex i is at PATH_SAMPLE_DIMENSION + 2 * i + 0
#define ANTI_ALIAS_SAMPLE_DIMENSION 0
#define LENS_SAMPLE_DIMENSION 2
#define PATH_SAMPLE_DIMENSION 4

vec3 randomDirection(int dimensionIndex) {
#ifdef BOUNCE
  // Compute a random direction in 3D
  vec2 ep = sample2(dimensionIndex);
  float ep0=ep[0];
  float ep1=ep[1];
  float theta = acos(2.0*ep0-1.0);
  float phi = ep1*2.0*M_PI;
  float x = sin(theta)*cos(phi);
  float y = sin(theta)*sin(phi);
  float z =cos(theta);
  return vec3(x,y,z);
  
#else
  return vec3(0);
#endif
}

vec3 getEmission(Material material, vec3 normal) {
#ifdef LIGHT  
  return material.emission;
#else
    // This is wrong. It just returns the diffuse color so that you see something to be sure it is working.
    return material.diffuse;
#endif
}

vec3 getReflectance(
  Material material,
  vec3 normal,
  vec3 inDirection,
  vec3 outDirection)
{
#ifdef THROUGHPUT    
]  //We calculate the specular and diffused component of BRDF in this function
  vec3 perfSpecDir = reflect(inDirection,normal);
  vec3 spec = material.specular * ( (2.0 + material.glossiness) / (2.0*M_PI) ) * pow(max(0.0, dot(outDirection, perfSpecDir)), material.glossiness);
  vec3 diffused = material.diffuse /(M_PI);
  return spec+diffused;
#else
  return vec3(1.0);
#endif 
}

vec3 getGeometricTerm(
  Material material,
  vec3 normal,
  vec3 inDirection,
  vec3 outDirection)
{
#ifdef THROUGHPUT  
]  //We calculate the geometric term in this function. 
  //Geometric term is cos theta where theta is angle between the normal and the outgoing Ray
  return vec3(max(0.0, dot( normalize(outDirection), normalize(normal) )));
#else
  return vec3(1.0);
#endif 
}

vec3 samplePath(Scene scene, Ray initialRay) {
  
  // Initial result is black
  vec3 result = vec3(0);
  
  Ray incomingRay = initialRay;
  vec3 throughput = vec3(1.0);
  
  for(int i = 0; i < maxPathLength; i++) {
    HitInfo hitInfo = intersectScene(scene, incomingRay, 0.001, 10000.0);  

    if(!hitInfo.hit) return result;
    result += throughput * getEmission(hitInfo.material, hitInfo.normal); 
    Ray outgoingRay;    

#ifdef BOUNCE
   // Code to compute the next ray
    //Calculate a new ray at the point of intersection. The origin of new ray is hit point.
    //Direction of new ray is random if using random sampling or determinintic if using Halton Sampling
    outgoingRay.direction = randomDirection(PATH_SAMPLE_DIMENSION+2*i+0);
    outgoingRay.origin = hitInfo.position;
    
    
#endif    
    //Probablity of every ray originating from centre of sphere and going towards sphere surface is equal.
    // So probability= 1/surface area of unit sphere = 1/4*pi
    float probability = 1.0/(4.0*M_PI);
#ifdef IMPORTANCE_SAMPLING
  // Put your code to importance-sample for the geometric term here
    // We check for the rays which have angle greater tha 90 degree with the normal.
    // We invert the direction of that ray , so that it goes towards brighter area.
    // We also increase the probability as now the probability of every ray to hit the upper hemisphere 
    //is 1/2*pi which is twice of 1/4*pi
    
     if(dot(outgoingRay.direction,hitInfo.normal)<0.0){
       outgoingRay.direction=-outgoingRay.direction;
       probability=probability*2.0;
     }
   
#endif

#ifdef THROUGHPUT    
    // Throughput computation which is combination of reflected and geometric component
    vec3 ref = getReflectance(hitInfo.material,hitInfo.normal,incomingRay.direction, outgoingRay.direction);
    vec3 geom = getGeometricTerm(hitInfo.material,hitInfo.normal,incomingRay.direction, outgoingRay.direction);

  throughput = throughput*ref*geom;
#else
    throughput *= 0.1;    
#endif
    
    throughput /= probability;
    
#ifdef BOUNCE
    // Handling of the next and the current ray here
    incomingRay = outgoingRay;
#endif    
  }  
  return result;
}

uniform ivec2 resolution;
Ray getFragCoordRay(vec2 fragCoord) {
  
    float sensorDistance = 1.0;
    vec3 origin = vec3(0, 0, sensorDistance);
    vec2 sensorMin = vec2(-1, -0.5);
    vec2 sensorMax = vec2(1, 0.5);
    vec2 pixelSize = (sensorMax - sensorMin) / vec2(resolution);
    vec3 direction = normalize(vec3(sensorMin + pixelSize * fragCoord, -sensorDistance));
  
    float apertureSize = 0.0;
    float focalPlane = 100.0;
    vec3 sensorPosition = origin + focalPlane * direction;  
    origin.xy += apertureSize * (sample2(LENS_SAMPLE_DIMENSION) - vec2(0.5));  
    direction = normalize(sensorPosition - origin);
  
    return Ray(origin, direction);
}

vec3 colorForFragment(Scene scene, vec2 fragCoord) {      
    initRandomSequence(); 
  
#ifdef AA  
  // Add anti aliasing code here
  // For doing anti aliasing, instead of shooting the ray from glFragCoord.xy , which is the centre of pixel,
  // I sample the rays through random positions in the pixel domian.
  // glFragCoord gives centre point of pixel. So I subtract the pixelWidth/1 and pixelHeight/2 form ite to get
  //bottom left coordinate of pixel. I then add a determinastically sampled no. between 0 and 1 to these coordinates
  // through which I pass the ray through the pixels.
  // In repeated iterations, due to selection of different points within the pixel, averaged color is given to pixel 
  // which address aliasing issues.
  // It is better than using a blurring kernel, as it smoothes out the value at every pixel by taking average of neighbouring pixels,
  // which in case of textured scene will be undesirable.
  vec2 sensorMin = vec2(-1, -0.5);
  vec2 sensorMax = vec2(1, 0.5);
  vec2 pixelSize = (sensorMax - sensorMin) / vec2(resolution);
  float sx = fragCoord.x- pixelSize.x/2.0 + sample(ANTI_ALIAS_SAMPLE_DIMENSION);
  float sy = fragCoord.y- pixelSize.x/2.0 + sample(ANTI_ALIAS_SAMPLE_DIMENSION+1);
    
  vec2 sampleCoord = vec2(sx,sy);
#else
  vec2 sampleCoord = fragCoord;
  
#endif
return samplePath(scene, getFragCoordRay(sampleCoord));
    
}


void loadScene1(inout Scene scene) {
  scene.spheres[0].position = vec3(1, -2, -12);
  scene.spheres[0].radius = 3.0;

  scene.spheres[0].material.diffuse = vec3(0.9, 0.1, 0.2);
  scene.spheres[0].material.specular = vec3(1.0);
  scene.spheres[0].material.glossiness = 10.0;
 

  scene.spheres[1].position = vec3(-8, -2, -12);
  scene.spheres[1].radius = 3.0;

  scene.spheres[1].material.diffuse = vec3(0.0);
  scene.spheres[1].material.specular = vec3(0.0);
  scene.spheres[1].material.glossiness = 10.0;
  scene.spheres[1].material.emission = 10.0*vec3(0.9,0.8,0.5);

  scene.planes[0].normal = vec3(0, 1, 0);
  scene.planes[0].d = 4.5;

  scene.planes[0].material.diffuse = vec3(0.8);
  scene.planes[0].material.specular = vec3(0);
  scene.planes[0].material.glossiness = 50.0;    

  scene.planes[1].normal = vec3(0, 0, 1);
  scene.planes[1].d = 18.5;

  scene.planes[1].material.diffuse = vec3(0.5, 0.8, 0.2);
  scene.planes[1].material.specular = vec3(0.0);
  scene.planes[1].material.glossiness = 5.0;
}

void main() {
  // Setup scene
  Scene scene;
  loadScene1(scene);

  // compute color for fragment
  gl_FragColor.rgb = colorForFragment(scene, gl_FragCoord.xy);
  gl_FragColor.a = 1.0;
}
