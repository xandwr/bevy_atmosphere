
struct Nishita {
    ray_origin: vec3<f32>,
    sun_position: vec3<f32>,
    sun_intensity: f32,
    planet_radius: f32,
    atmosphere_radius: f32,
    rayleigh_coefficient: vec3<f32>,
    rayleigh_scale_height: f32,
    mie_coefficient: f32,
    mie_scale_height: f32,
    mie_direction: f32,
    moon_intensity: f32,
}

const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const ISTEPS: u32 = 16u;
const JSTEPS: u32 = 8u;

// Star field constants
const STAR_COUNT: u32 = 1500u;
const STAR_SEED: u32 = 0x9E3779B9u; // Golden ratio based seed for good distribution

// Hash function for deterministic random numbers (based on PCG)
fn hash_u32(state: u32) -> u32 {
    var s = state;
    s = s * 747796405u + 2891336453u;
    s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (s >> 22u) ^ s;
}

// Convert hash to float in [0, 1)
fn hash_to_float(h: u32) -> f32 {
    return f32(h) / 4294967296.0;
}

// Generate a deterministic random direction on a sphere for star i
fn star_direction(i: u32) -> vec3<f32> {
    let h1 = hash_u32(i ^ STAR_SEED);
    let h2 = hash_u32(h1);

    // Use spherical coordinates with uniform distribution
    let theta = hash_to_float(h1) * TAU;
    let phi = acos(2.0 * hash_to_float(h2) - 1.0);

    return vec3<f32>(
        sin(phi) * cos(theta),
        sin(phi) * sin(theta),
        cos(phi)
    );
}

// Get star brightness (varied for visual interest)
fn star_brightness(i: u32) -> f32 {
    let h = hash_u32(i ^ 0x85EBCA6Bu); // Different seed for brightness
    let base = hash_to_float(h);
    // Power distribution: many dim stars, few bright ones
    return pow(base, 2.0) * 0.8 + 0.2;
}

// Get star color temperature variation
fn star_color(i: u32) -> vec3<f32> {
    let h = hash_u32(i ^ 0xC2B2AE35u); // Different seed for color
    let temp = hash_to_float(h);

    // Vary from blue-white to yellow-white
    if temp < 0.3 {
        // Blue-white stars
        return vec3<f32>(0.8, 0.85, 1.0);
    } else if temp < 0.7 {
        // White stars
        return vec3<f32>(1.0, 1.0, 1.0);
    } else {
        // Yellow-white stars
        return vec3<f32>(1.0, 0.95, 0.8);
    }
}

// Render stars for a given ray direction
fn render_stars(ray: vec3<f32>, sun_y: f32) -> vec3<f32> {
    // Night factor: stars only visible when sun is below horizon
    // Fade in between sun_y = 0.1 and sun_y = -0.2
    let night_factor = smoothstep(0.1, -0.2, sun_y);

    if night_factor <= 0.0 {
        return vec3<f32>(0.0);
    }

    let r = normalize(ray);
    var star_contribution = vec3<f32>(0.0);

    // Angular size of stars (in radians, very small for crisp points)
    let star_radius = 0.0012;

    for (var i = 0u; i < STAR_COUNT; i++) {
        let star_dir = star_direction(i);

        // Skip stars below horizon
        if star_dir.y < -0.1 {
            continue;
        }

        // Calculate angular distance to this star
        let cos_angle = dot(r, star_dir);

        // Quick rejection for stars not in view
        if cos_angle < 0.99 {
            continue;
        }

        let angle = acos(clamp(cos_angle, -1.0, 1.0));

        // Star point with soft falloff
        if angle < star_radius {
            let brightness = star_brightness(i);
            let color = star_color(i);

            // Soft circular falloff
            let falloff = 1.0 - (angle / star_radius);
            let intensity = falloff * falloff * brightness;

            star_contribution += color * intensity * 2.0;
        }
    }

    return star_contribution * night_factor;
}

fn rsi(rd: vec3<f32>, r0: vec3<f32>, sr: f32) -> vec2<f32> {
    // ray-sphere intersection that assumes
    // the sphere is centered at the origin.
    // No intersection when result.x > result.y
    let a = dot(rd, rd);
    let b = 2.0 * dot(rd, r0);
    let c = dot(r0, r0) - (sr * sr);
    let d = (b * b) - (4.0 * a * c);

    if d < 0.0 {
        return vec2<f32>(1e5, -1e5);
    } else {
        return vec2<f32>(
            (-b - sqrt(d)) / (2.0 * a),
            (-b + sqrt(d)) / (2.0 * a)
        );
    }
}

fn render_nishita(r_full: vec3<f32>, r0: vec3<f32>, p_sun_full: vec3<f32>, i_sun: f32, r_planet: f32, r_atmos: f32, k_rlh: vec3<f32>, k_mie: f32, sh_rlh: f32, sh_mie: f32, g: f32, i_moon: f32) -> vec3<f32> {
    // Normalize the ray direction and sun position.
    let r = normalize(r_full);
    let p_sun = normalize(p_sun_full);

    // Calculate the step size of the primary ray.
    var p = rsi(r, r0, r_atmos);
    if p.x > p.y { return vec3<f32>(0f); }
    p.y = min(p.y, rsi(r, r0, r_planet).x);
    let i_step_size = (p.y - p.x) / f32(ISTEPS);

    // Initialize the primary ray depth.
    var i_depth = 0.0;

    // Initialize accumulators for Rayleigh and Mie scattering.
    var total_rlh = vec3<f32>(0f);
    var total_mie = vec3<f32>(0f);

    // Initialize optical depth accumulators for the primary ray.
    var i_od_rlh = 0f;
    var i_od_mie = 0f;

    // Calculate the Rayleigh and Mie phases.
    let mu = dot(r, p_sun);
    let mumu = mu * mu;
    let gg = g * g;
    let p_rlh = 3.0 / (16.0 * PI) * (1.0 + mumu);
    let p_mie = 3.0 / (8.0 * PI) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));

    // Sample the primary ray.
    for (var i = 0u; i < ISTEPS; i++) {
        // Calculate the primary ray sample position.
        let i_pos = r0 + r * (i_depth + i_step_size * 0.5);

        // Calculate the height of the sample.
        let i_height = length(i_pos) - r_planet;

        // Calculate the optical depth of the Rayleigh and Mie scattering for this step.
        let od_step_rlh = exp(-i_height / sh_rlh) * i_step_size;
        let od_step_mie = exp(-i_height / sh_mie) * i_step_size;

        // Accumulate optical depth.
        i_od_rlh += od_step_rlh;
        i_od_mie += od_step_mie;

        // Calculate the step size of the secondary ray.
        let j_step_size = rsi(p_sun, i_pos, r_atmos).y / f32(JSTEPS);

        // Initialize the secondary ray depth.
        var j_depth = 0f;

        // Initialize optical depth accumulators for the secondary ray.
        var j_od_rlh = 0f;
        var j_od_mie = 0f;

        // Sample the secondary ray.
        for (var j = 0u; j < JSTEPS; j++) {

            // Calculate the secondary ray sample position.
            let j_pos = i_pos + p_sun * (j_depth + j_step_size * 0.5);

            // Calculate the height of the sample.
            let j_height = length(j_pos) - r_planet;

            // Accumulate the optical depth.
            j_od_rlh += exp(-j_height / sh_rlh) * j_step_size;
            j_od_mie += exp(-j_height / sh_mie) * j_step_size;

            // Increment the secondary ray depth.
            j_depth += j_step_size;
        }

        // Calculate attenuation.
        let attn = exp(-(k_mie * (i_od_mie + j_od_mie) + k_rlh * (i_od_rlh + j_od_rlh)));

        // Accumulate scattering.
        total_rlh += od_step_rlh * attn;
        total_mie += od_step_mie * attn;

        // Increment the primary ray depth.
        i_depth += i_step_size;
    }

    // Calculate atmospheric scattering color
    let atmosphere_color = i_sun * (p_rlh * k_rlh * total_rlh + p_mie * k_mie * total_mie);

    // Add sun disc rendering
    // Calculate the angular distance from the ray to the sun
    let sun_angular_radius = 0.0045; // Sun's angular size (about 0.53 degrees in radians)
    let sun_dot = dot(r, p_sun);
    let sun_angle = acos(clamp(sun_dot, -1.0, 1.0));

    // Create a sharp sun disc with soft edge
    var sun_disc = 0.0;
    if sun_angle < sun_angular_radius {
        // Core of the sun - full brightness
        sun_disc = 1.0;
    } else if sun_angle < sun_angular_radius * 1.5 {
        // Soft edge falloff
        let edge_factor = (sun_angular_radius * 1.5 - sun_angle) / (sun_angular_radius * 0.5);
        sun_disc = smoothstep(0.0, 1.0, edge_factor);
    }

    // Sun color (warm yellow-white)
    let sun_color = vec3<f32>(1.0, 0.95, 0.85) * i_sun * 20.0;

    // Blend sun disc with atmosphere
    var final_color = atmosphere_color + sun_color * sun_disc;

    // Moon rendering (opposite the sun)
    if i_moon > 0.0 {
        let p_moon = -p_sun; // Moon is opposite the sun
        let moon_angular_radius = 0.0046; // Slightly larger than sun for visual interest
        let moon_dot = dot(r, p_moon);
        let moon_angle = acos(clamp(moon_dot, -1.0, 1.0));

        var moon_disc = 0.0;
        if moon_angle < moon_angular_radius {
            // Core of the moon - full brightness
            moon_disc = 1.0;
        } else if moon_angle < moon_angular_radius * 1.3 {
            // Soft edge falloff (slightly sharper than sun)
            let edge_factor = (moon_angular_radius * 1.3 - moon_angle) / (moon_angular_radius * 0.3);
            moon_disc = smoothstep(0.0, 1.0, edge_factor);
        }

        // Moon color (pale silvery white with slight blue tint)
        let moon_color = vec3<f32>(0.85, 0.88, 0.95) * i_moon;

        // Add subtle moon glow
        let glow_radius = moon_angular_radius * 4.0;
        var moon_glow = 0.0;
        if moon_angle < glow_radius {
            moon_glow = pow(1.0 - moon_angle / glow_radius, 2.0) * 0.15;
        }
        let glow_color = vec3<f32>(0.7, 0.75, 0.85) * i_moon * moon_glow;

        final_color += moon_color * moon_disc + glow_color;
    }

    // Add procedural stars (only visible at night)
    let stars = render_stars(r, p_sun.y);
    final_color += stars;

    return final_color;
}

@group(0) @binding(0)
var<uniform> nishita: Nishita;

@group(1) @binding(0)
var image: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let size = textureDimensions(image).x;
    let scale = f32(size) / 2f;

    let dir = vec2<f32>((f32(invocation_id.x) / scale) - 1f, (f32(invocation_id.y) / scale) - 1f);

    var ray: vec3<f32>;

    switch invocation_id.z {
        case 0u {
            ray = vec3<f32>(1f, -dir.y, -dir.x); // +X
        }
        case 1u {
            ray = vec3<f32>(-1f, -dir.y, dir.x);// -X
        }
        case 2u {
            ray = vec3<f32>(dir.x, 1f, dir.y); // +Y
        }
        case 3u {
            ray = vec3<f32>(dir.x, -1f, -dir.y);// -Y
        }
        case 4u {
            ray = vec3<f32>(dir.x, -dir.y, 1f); // +Z
        }
        default: {
            ray = vec3<f32>(-dir.x, -dir.y, -1f);// -Z
        }
    }

    let render = render_nishita(
        ray,
        nishita.ray_origin,
        nishita.sun_position,
        nishita.sun_intensity,
        nishita.planet_radius,
        nishita.atmosphere_radius,
        nishita.rayleigh_coefficient,
        nishita.mie_coefficient,
        nishita.rayleigh_scale_height,
        nishita.mie_scale_height,
        nishita.mie_direction,
        nishita.moon_intensity,
    );

    textureStore(
        image,
        vec2<i32>(invocation_id.xy),
        i32(invocation_id.z),
        vec4<f32>(render, 1.0)
    );
}
